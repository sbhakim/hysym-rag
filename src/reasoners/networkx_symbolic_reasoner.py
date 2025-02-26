# src/reasoners/networkx_symbolic_reasoner.py

import json
import spacy
import networkx as nx
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn  # For potential adapter usage in fallback
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from collections import defaultdict
from datetime import datetime

from src.utils.device_manager import DeviceManager
from src.utils.dimension_manager import DimensionalityManager

# For improved keyword extraction using KeyBERT
from keybert import KeyBERT

kw_model = KeyBERT()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GraphSymbolicReasoner:
    """
    Enhanced graph-based symbolic reasoner for academic evaluation of HySym-RAG.
    Core functionalities include loading and validating rules, building a knowledge graph,
    indexing rules, matching rules against query embeddings, and basic query processing.
    Detailed reasoning chain extraction and academic metrics are moved to a separate module.
    """

    def __init__(
            self,
            rules_file: str,
            match_threshold: float = 0.1,  # Reduced match_threshold
            max_hops: int = 5,
            embedding_model: str = 'all-MiniLM-L6-v2',
            device: Optional[torch.device] = None,
            dim_manager: Optional[DimensionalityManager] = None
    ):
        self.logger = logger
        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.device = device or DeviceManager.get_device()
        self.dim_manager = dim_manager

        # Initialize embedder
        self.embedder = SentenceTransformer(embedding_model).to(self.device)

        # Load spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            raise

        # Load and validate rules
        self.rules = self._load_rules(rules_file)

        # Basic academic metrics
        self.reasoning_metrics = {
            'path_lengths': [],
            'match_confidences': [],
            'hop_distributions': defaultdict(int),
            'pattern_types': defaultdict(int),
            'chains': []
        }

        # Knowledge graph and indexes
        self.graph = nx.DiGraph()
        self.rule_index: Dict[str, Dict] = {}
        self.rule_ids: List[str] = []
        self.keyword_index = defaultdict(list)
        self.entity_index = defaultdict(set)
        self.relation_index = defaultdict(set)
        self.rule_embeddings = None

        # Build indexes and graph
        self.build_rule_index()
        self.build_graph()

        logger.info("GraphSymbolicReasoner initialized successfully")

    def _load_rules(self, rules_file: str) -> Dict[str, Dict]:
        loaded_rules = {}
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                raw_rules = json.load(f)
            self.logger.info(f"Successfully loaded {len(raw_rules)} rules from {rules_file}")
            for i, rule in enumerate(raw_rules):
                if self._validate_rule_structure(rule):
                    rule_id = f"rule_{i}"
                    rule['version'] = GraphSymbolicReasoner._get_next_version(rule_id, loaded_rules)
                    loaded_rules[rule_id] = rule
                else:
                    self.logger.warning(f"Invalid rule structure skipped: {rule}")
            return loaded_rules
        except FileNotFoundError:
            self.logger.warning(f"Rules file not found at {rules_file}. Starting with empty rule set.")
            return {}
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON from {rules_file}. Please ensure it's valid JSON.")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading rules from {rules_file}: {str(e)}")
            return {}

    @staticmethod
    def _get_next_version(rule_id: str, current_rules: Dict[str, Dict]) -> int:
        return sum(1 for r in current_rules.values() if r.get('id', '').startswith(rule_id)) + 1

    def build_rule_index(self):
        self.rule_index = {}
        valid_rule_ids = []
        rule_embeddings_list = []
        embedding_dim = None

        for rule_id, rule in self.rules.items():
            self.rule_index[rule_id] = rule

            if 'keywords' in rule:
                for keyword in rule['keywords']:
                    self.keyword_index[keyword].append(rule_id)

            try:
                # Use existing embedding if present
                if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                    rule_embedding = rule['embedding'].unsqueeze(0) if rule['embedding'].dim() == 1 else rule[
                        'embedding']
                # Or compute from source_text
                elif 'source_text' in rule and rule['source_text'].strip():
                    rule_text = rule['source_text']
                    if 'keywords' in rule:
                        rule_text += " " + " ".join(rule['keywords'])
                    rule_embedding = self.embedder.encode(rule_text, convert_to_tensor=True).to(self.device)
                    rule_embedding = rule_embedding.unsqueeze(0) if rule_embedding.dim() == 1 else rule_embedding
                # Or compute from response
                elif 'response' in rule and rule['response'].strip():
                    rule_text = rule['response']
                    if 'keywords' in rule:
                        rule_text += " " + " ".join(rule['keywords'])
                    rule_embedding = self.embedder.encode(rule_text, convert_to_tensor=True).to(self.device)
                    rule_embedding = rule_embedding.unsqueeze(0) if rule_embedding.dim() == 1 else rule_embedding
                else:
                    continue

                # Use the centralized alignment from DimensionalityManager with logging
                current_embedding = self.dim_manager.align_embeddings(rule_embedding.to(self.device), "rule")
                if embedding_dim is None:
                    embedding_dim = current_embedding.shape[-1]
                elif current_embedding.shape[-1] != embedding_dim:
                    self.logger.warning(
                        f"Rule {rule_id} embedding dimension mismatch. Expected {embedding_dim}, got {current_embedding.shape[-1]}"
                    )
                    continue

                if current_embedding.shape[-1] != self.dim_manager.target_dim:
                    self.logger.warning(
                        f"Rule {rule_id} embedding dimension {current_embedding.shape[-1]} does not match target {self.dim_manager.target_dim}"
                    )
                    continue

                rule['embedding'] = current_embedding
                valid_rule_ids.append(rule_id)
                rule_embeddings_list.append(current_embedding)
            except Exception as e:
                self.logger.error(f"Error processing rule {rule_id}: {e}")
                continue

        self.rule_ids = valid_rule_ids
        if rule_embeddings_list:
            try:
                stacked_embeddings = torch.cat(rule_embeddings_list).to(self.device)
                if stacked_embeddings.shape[-1] != self.dim_manager.target_dim:
                    self.logger.warning("Correcting rule embeddings dimension...")
                    stacked_embeddings = self.dim_manager.align_embeddings(stacked_embeddings, "rule")
                self.rule_embeddings = stacked_embeddings
                self.logger.info(f"Rule embeddings tensor created with shape: {self.rule_embeddings.shape}")
            except Exception as e:
                self.logger.error(f"Error stacking rule embeddings: {e}")
                self.rule_embeddings = None
        else:
            self.rule_embeddings = None

        self.logger.info(
            f"Rule index built successfully with {len(self.rules)} rules. Final count of valid rules: {len(self.rule_ids)}"
        )

    def build_graph(self):
        try:
            self.graph = nx.DiGraph()
            for rule_id, rule in self.rule_index.items():
                self.graph.add_node(
                    rule_id,
                    type="fact",
                    rule_response=rule.get('response', ''),
                    confidence=rule.get('confidence', 0.5),
                    version=rule.get('version', 0),
                    keywords=rule.get('keywords', [])
                )
                self._add_rule_relationships(rule_id, rule)
            self.logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes")
        except Exception as e:
            self.logger.error(f"Error building knowledge graph: {str(e)}")
            self.graph = nx.DiGraph()

    def _add_rule_relationships(self, rule_id: str, rule: Dict) -> None:
        for other_id in self.graph.nodes:
            if other_id != rule_id:
                other_rule = self.graph.nodes[other_id]
                common_keywords = set(rule.get('keywords', [])) & set(other_rule.get('keywords', []))
                if common_keywords:
                    relationship_strength = len(common_keywords) / max(
                        len(rule.get('keywords', [])),
                        len(other_rule.get('keywords', []))
                    )
                    if relationship_strength > self.match_threshold:
                        self.graph.add_edge(
                            rule_id,
                            other_id,
                            weight=relationship_strength,
                            common_keywords=list(common_keywords)
                        )

    def _validate_rule_structure(self, rule: Dict) -> bool:
        try:
            if not isinstance(rule, dict):
                return False

            if "supporting_fact" in rule:
                required_hotpot_fields = {"type", "source_text", "keywords"}
                if all(field in rule and isinstance(rule[field], (str, list)) for field in required_hotpot_fields):
                    if "confidence" not in rule:
                        rule["confidence"] = self._calculate_rule_confidence(rule)
                    return True
                else:
                    return False

            required_fields = {"keywords", "response"}
            if all(field in rule and isinstance(rule[field], (str, list)) for field in required_fields):
                if "embedding" in rule:
                    return isinstance(rule["embedding"], torch.Tensor)
                if ("source_text" in rule and isinstance(rule["source_text"], str)
                        and rule["source_text"].strip()):
                    return True
                if ("response" in rule and isinstance(rule["response"], str)
                        and rule["response"].strip()):
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error in rule validation: {str(e)}")
            return False

    def _calculate_rule_confidence(self, rule: Dict) -> float:
        base_confidence = 0.7
        if "source_text" in rule:
            base_confidence += 0.1
        if "entity_types" in rule:
            base_confidence += 0.1
        if "type" in rule:
            base_confidence += 0.1
        return min(1.0, base_confidence)

    def add_dynamic_rules(self, new_rules: List[Dict]) -> None:
        if not new_rules:
            self.logger.info("No new rules to add.")
            return

        valid_rules = {}
        for i, rule in enumerate(new_rules, start=len(self.rules)):
            standardized_rule = self._standardize_rule_format(rule)
            if self._validate_rule_structure(standardized_rule):
                rule_id = f"rule_{i}"
                standardized_rule['id'] = rule_id
                standardized_rule['version'] = GraphSymbolicReasoner._get_next_version(rule_id, self.rules)
                standardized_rule['added_timestamp'] = datetime.now().isoformat()
                if 'confidence' not in standardized_rule:
                    standardized_rule['confidence'] = self._calculate_rule_confidence(standardized_rule)
                if 'embedding' in standardized_rule and isinstance(standardized_rule['embedding'], torch.Tensor):
                    standardized_rule['embedding'] = standardized_rule['embedding'].to(self.device)
                valid_rules[rule_id] = standardized_rule
            else:
                self.logger.warning(f"Invalid rule structure (post-standardization): {rule}")

        if valid_rules:
            self.rules.update(valid_rules)
            self.build_rule_index()
            self.build_graph()
            self.logger.info(f"Added {len(valid_rules)} new rules. Total rules: {len(self.rules)}")
            try:
                self._track_rule_addition(list(valid_rules.values()))
            except Exception as e:
                self.logger.error(f"Error tracking rule addition (non-critical): {str(e)}")

    def _standardize_rule_format(self, rule: Dict) -> Dict:
        standardized = rule.copy()

        if "supporting_fact" in standardized and "source_text" in standardized and "response" not in standardized:
            standardized["response"] = standardized["source_text"]

        if "statement" in standardized and "response" not in standardized:
            standardized["response"] = standardized["statement"]
        elif "text" in standardized and "response" not in standardized:
            standardized["response"] = standardized["text"]

        if "embedding" in standardized and "keywords" not in standardized:
            text_to_process = standardized.get("response", standardized.get("source_text", ""))
            keywords = self._extract_keywords_from_text(text_to_process)
            if keywords:
                standardized["keywords"] = keywords

        if "keywords" in standardized and isinstance(standardized["keywords"], str):
            standardized["keywords"] = [kw.strip() for kw in standardized["keywords"].split(',')]

        return standardized

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        try:
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
            return [kw[0] for kw in keywords]
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return [word.lower() for word in text.split() if len(word) > 3]

    def _track_rule_addition(self, valid_rules: List[Dict]):
        num_rules_added = len(valid_rules)
        total_rules_now = len(self.rules)
        avg_confidence = float(np.mean([rule.get('confidence', 0.0) for rule in valid_rules])) if valid_rules else 0.0
        self.logger.info(
            f"Added {num_rules_added} rules. Total rules: {total_rules_now}. Average confidence: {avg_confidence:.3f}"
        )
        if 'rule_additions' in self.reasoning_metrics:
            for rule in valid_rules:
                self.reasoning_metrics['rule_additions'].append({
                    'timestamp': datetime.now().isoformat(),
                    'confidence': rule.get('confidence', 0.0),
                    'type': rule.get('type', 'unknown')
                })

    def _calculate_similarity(self, query_embedding: torch.Tensor, rule_embedding: torch.Tensor) -> float:
        """
        Directly calculates cosine similarity after alignment using the centralized alignment functions.
        """
        try:
            query_aligned = self.dim_manager.align_embeddings(query_embedding, "query")
            rule_aligned = self.dim_manager.align_embeddings(rule_embedding, "rule")

            query_aligned = query_aligned.to(self.device)
            rule_aligned = rule_aligned.to(self.device)

            self.logger.debug(f"Aligned shapes - Query: {query_aligned.shape}, Rule: {rule_aligned.shape}")

            with torch.no_grad():
                similarity = F.cosine_similarity(query_aligned, rule_aligned, dim=1)
                return similarity.item()
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def process_query(self, query: Union[str, torch.Tensor]) -> Dict:
        """
        Attempt fast direct matching; if no match, do multi-hop.
        Delegates all dimension alignment to DimensionalityManager with detailed logging.
        """
        try:
            if isinstance(query, str):
                logger.debug(f"Received string query: {query}, encoding to tensor")
                query_embedding_raw = self.embedder.encode(query, convert_to_tensor=True).to(self.device)
            elif isinstance(query, torch.Tensor):
                query_embedding_raw = query.to(self.device)
            else:
                raise ValueError(f"Unsupported query type: {type(query)}")

            logger.debug(f"Raw query embedding shape: {query_embedding_raw.shape}")

            # Use centralized alignment with logging
            query_aligned = self.dim_manager.align_embeddings(query_embedding_raw, "query").view(1, -1)
            logger.debug(f"Aligned query shape: {query_aligned.shape}")

            rules_aligned = self.dim_manager.align_embeddings(self.rule_embeddings.clone().detach(), "rule")
            logger.debug(f"Aligned rule embeddings shape: {rules_aligned.shape}")

            if query_aligned.shape[1] != rules_aligned.shape[1]:
                raise ValueError(
                    f"Dimension mismatch: query embedding dim {query_aligned.shape[1]} != rule embedding dim {rules_aligned.shape[1]}"
                )

            # Direct matching using cosine similarity
            similarities = util.cos_sim(query_aligned, rules_aligned.squeeze(1)).flatten()
            matching_indices = (similarities >= self.match_threshold).nonzero(as_tuple=False).flatten().tolist()

            if matching_indices:
                # Sort matches by similarity and take top N most relevant
                sorted_indices = sorted(
                    matching_indices,
                    key=lambda idx: similarities[idx].item(),
                    reverse=True
                )[:5]  # Limit to top 5 most relevant matches

                responses = []
                similarity_scores = []

                for idx in sorted_indices:
                    rule_id = self.rule_ids[idx]
                    rule = self.rules.get(rule_id, {})
                    sim_score = similarities[idx].item()

                    # Only include rules with reasonable similarity
                    if "response" in rule and sim_score >= self.match_threshold * 1.2:
                        # Check if response is relevant to query
                        if self._check_rule_relevance(rule.get("response", ""), query):
                            responses.append(rule["response"])
                            similarity_scores.append(sim_score)

                if responses:
                    logger.info(f"Found {len(responses)} relevant symbolic responses.")
                    return {
                        "response": responses,
                        "similarities": similarity_scores,
                        "preamble": f"Relevant background for: {query if isinstance(query, str) else 'the query'}"
                    }
                else:
                    logger.info("No highly relevant symbolic matches found.")
                    return {"response": ["No highly relevant symbolic match found."]}
            else:
                logger.info("No symbolic match found via direct similarity.")

            # Multi-hop fallback: graph traversal
            multi_hop_paths = self._some_graph_traversal(query_aligned)
            all_hop_embeddings = []
            for rule_id in multi_hop_paths:
                rule = self.rules.get(rule_id, {})
                if "embedding" in rule:
                    hop_emb = rule["embedding"]
                    all_hop_embeddings.append(hop_emb.unsqueeze(0))
                else:
                    logger.warning(f"Rule {rule_id} missing 'embedding' key in multi-hop traversal.")

            if all_hop_embeddings:
                multi_hop_emb = torch.cat(all_hop_embeddings, dim=0).unsqueeze(0)
            else:
                multi_hop_emb = query_aligned.unsqueeze(0)

            responses = self.traverse_graph_from_multi_hop(multi_hop_emb)
            chain_info = {'steps': responses, 'reasoning_path': responses}
            self.reasoning_metrics['chains'].append(chain_info)

            self._update_reasoning_metrics(responses)

            if not responses:
                logger.info("No symbolic match found in multi-hop traversal.")
                return {"response": ["No symbolic match found."]}
            logger.info(f"Found {len(responses)} symbolic responses via multi-hop traversal.")
            return {"response": responses}

        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return {"response": [f"Error processing query: {str(e)}"]}

    def _some_graph_traversal(self, query_embedding: torch.Tensor) -> List[str]:
        """
        Combined approach for graph traversal.
        Uses centralized alignment to log and fix dimensions before applying matrix multiplication.
        """
        if not self.graph.nodes or self.rule_embeddings is None:
            return []

        try:
            self.logger.debug(f"Initial query shape: {query_embedding.shape}")
            self.logger.debug(f"Initial rule embeddings shape: {self.rule_embeddings.shape}")

            query_aligned = self.dim_manager.align_embeddings(query_embedding.clone(), "query")
            rules_aligned = self.dim_manager.align_embeddings(self.rule_embeddings.clone().detach(), "rule")
            self.logger.debug(f"Aligned query shape: {query_aligned.shape}")
            self.logger.debug(f"Aligned rule embeddings shape: {rules_aligned.shape}")

            self.logger.debug(f"Using matrix multiplication with shapes: {query_aligned.shape} x {rules_aligned.shape}")
            similarities = torch.matmul(query_aligned, rules_aligned.transpose(0, 1)).squeeze()
            self.logger.debug(f"Matrix multiplication successful, similarity shape: {similarities.shape}")

        except RuntimeError as e:
            self.logger.warning(f"Matrix multiplication failed: {e}. Falling back to element-wise calculation.")
            similarities = torch.zeros(len(self.rule_ids), device=self.device)
            for idx, rule_id in enumerate(self.rule_ids):
                try:
                    rule = self.rules.get(rule_id, {})
                    if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                        rule_emb = rule['embedding'].to(self.device)
                        q_flat = query_aligned.view(-1)
                        r_flat = rule_emb.view(-1)
                        if q_flat.shape[0] != r_flat.shape[0]:
                            self.logger.warning(
                                f"Dimension mismatch in fallback for rule {rule_id}. Aligning manually.")
                            q_flat = self.dim_manager.align_embeddings(q_flat.unsqueeze(0), "query").view(-1)
                            r_flat = self.dim_manager.align_embeddings(r_flat.unsqueeze(0), "rule").view(-1)
                        dot_product = torch.dot(q_flat, r_flat)
                        norm_q = torch.norm(q_flat)
                        norm_r = torch.norm(r_flat)
                        sim = dot_product / (norm_q * norm_r)
                        similarities[idx] = sim
                except Exception as inner_e:
                    self.logger.warning(f"Error calculating similarity for rule {rule_id}: {inner_e}")
                    similarities[idx] = 0.0

        if similarities.numel() == 0:
            return []

        # Breadth-first search from best matching rule
        start_idx = torch.argmax(similarities).item()
        start_node = self.rule_ids[start_idx]
        bfs_tree = nx.bfs_tree(self.graph, source=start_node, depth_limit=self.max_hops)
        return list(bfs_tree.nodes())

    def traverse_graph_from_multi_hop(self, multi_hop_emb: torch.Tensor) -> List[str]:
        responses = []
        hop_count = multi_hop_emb.size(1)
        for i in range(hop_count):
            if i < len(self.rule_ids):
                rule_id = self.rule_ids[i]
                rule = self.rules.get(rule_id, {})
                responses.append(rule.get('response', ''))
        return responses

    def _update_reasoning_metrics(self, responses: List[str]):
        chain_length = len(responses) if responses else 1
        self.reasoning_metrics['path_lengths'].append(chain_length)
        confidences = [self._calculate_response_confidence(response) for response in responses]
        self.reasoning_metrics['match_confidences'].extend(confidences)
        for response in responses:
            rules_used = self._identify_rules_used(response)
            for rule_id in rules_used:
                self.reasoning_metrics.setdefault('rule_utilization', defaultdict(int))[rule_id] += 1

    def _calculate_response_confidence(self, response: str) -> float:
        confidence_factors = []
        length_score = min(len(response.split()) / 50, 1.0)
        confidence_factors.append(length_score)
        doc = self.nlp(response)
        structure_score = len([ent for ent in doc.ents]) / max(1, len(response.split()))
        confidence_factors.append(structure_score)
        return float(np.mean(confidence_factors))

    def _check_rule_relevance(self, rule_text: str, query: str) -> bool:
        """Check if a rule is semantically relevant to the query."""
        if isinstance(query, torch.Tensor):
            return True  # Skip relevance check for tensor queries

        # Extract key terms from query and rule
        import re
        query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
        rule_terms = set(re.findall(r'\b\w{4,}\b', rule_text.lower()))

        # Check for overlap in key terms
        common_terms = query_terms.intersection(rule_terms)
        if len(common_terms) >= min(2, len(query_terms) / 3):
            return True

        # Fallback to embedding similarity for complex queries
        try:
            query_emb = self.embedder.encode(query, convert_to_tensor=True)
            rule_emb = self.embedder.encode(rule_text, convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(query_emb, rule_emb, dim=0).item()
            return similarity > 0.4  # Higher threshold for semantic similarity
        except:
            # If embedding comparison fails, be permissive
            return len(query_terms) < 3 or len(rule_terms) < 3

    def _identify_rules_used(self, response: str) -> Set[str]:
        used_rules = set()
        response_embedding = self.embedder.encode(response, convert_to_tensor=True).to(self.device)
        for rule_id, rule in self.rules.items():
            if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                rule_embedding = rule['embedding'].to(self.device)
                similarity = util.cos_sim(response_embedding, rule_embedding).item()
                if similarity > self.match_threshold:
                    used_rules.add(rule_id)
        return used_rules

    def _log_memory_usage(self, stage: str):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Memory usage at {stage}: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    def _ensure_device(self, tensor: torch.Tensor, target_device: Optional[torch.device] = None) -> torch.Tensor:
        device = target_device or self.device
        moved_tensor, _ = DeviceManager.ensure_same_device(tensor, tensor, device=device)
        return moved_tensor
