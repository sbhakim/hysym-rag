# src/reasoners/networkx_symbolic_reasoner.py

import json
import spacy
import networkx as nx
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime
from src.utils.device_manager import DeviceManager
from src.utils.dimension_manager import DimensionalityManager

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

    def __init__(self,
                 rules_file: str,
                 match_threshold: float = 0.1,  # Reduced match_threshold
                 max_hops: int = 5,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 device: Optional[torch.device] = None,
                 dim_manager: Optional[DimensionalityManager] = None  # Add DimensionalityManager
                 ):
        """
        Initialize the symbolic reasoner with proper tensor handling and basic academic metrics tracking.
        """
        self.logger = logger

        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.device = device or DeviceManager.get_device()
        self.dim_manager = dim_manager  # Store DimensionalityManager

        # Initialize embedder with device specification
        self.embedder = SentenceTransformer(embedding_model).to(self.device)
        # Load spaCy model for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            raise

        # Load and validate rules into a dictionary keyed by rule ID
        self.rules = self._load_rules(rules_file)

        # Initialize academic tracking metrics (basic ones only; detailed chain extraction is handled elsewhere)
        self.reasoning_metrics = {
            'path_lengths': [],
            'match_confidences': [],
            'hop_distributions': defaultdict(int),
            'pattern_types': defaultdict(int),
            'chains': []  # New key to store reasoning chain info
        }

        # Initialize knowledge graph and related indexes
        self.graph = nx.DiGraph()  # The knowledge graph (DAG)
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
        """
        Load rules from a JSON file with enhanced validation and versioning.
        Returns a dictionary keyed by a generated rule ID.
        """
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
        """
        Get the next version number for a rule.
        """
        return sum(1 for r in current_rules.values() if r.get('id', '').startswith(rule_id)) + 1

    def build_rule_index(self):
        """
        Build an index for rules to speed up retrieval.
        This updated method validates embedding dimensions before stacking.
        """
        self.rule_index = {}
        valid_rule_ids = []
        rule_embeddings_list = []
        embedding_dim = None

        for rule_id, rule in self.rules.items():
            # Add rule to index regardless; we'll only add valid embeddings to the parallel lists.
            self.rule_index[rule_id] = rule

            if 'keywords' in rule:
                for keyword in rule['keywords']:
                    self.keyword_index[keyword].append(rule_id)

            try:
                if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                    self.logger.debug(f"Rule ID: {rule_id} Original rule embedding shape: {rule['embedding'].shape}")
                    # Ensure the tensor is 2D
                    if rule['embedding'].dim() == 1:
                        rule_embedding = rule['embedding'].unsqueeze(0)
                    else:
                        rule_embedding = rule['embedding']

                    current_embedding = self.dim_manager.align_embeddings(rule_embedding.to(self.device), "rule")
                    self.logger.debug(f"Rule ID: {rule_id} Aligned rule embedding shape: {current_embedding.shape}")

                    # Set embedding_dim if not already set; otherwise, validate consistency
                    if embedding_dim is None:
                        embedding_dim = current_embedding.shape[-1]
                    elif current_embedding.shape[-1] != embedding_dim:
                        self.logger.warning(
                            f"Rule {rule_id} embedding dimension mismatch. Expected {embedding_dim}, got {current_embedding.shape[-1]}"
                        )
                        continue

                    # Validate final target dimension
                    if current_embedding.shape[-1] != self.dim_manager.target_dim:
                        self.logger.warning(
                            f"Rule {rule_id} embedding dimension {current_embedding.shape[-1]} does not match target {self.dim_manager.target_dim}")
                        continue

                    rule['embedding'] = current_embedding
                    valid_rule_ids.append(rule_id)
                    rule_embeddings_list.append(current_embedding)
                elif 'source_text' in rule:
                    rule_embedding = self.embedder.encode(rule['source_text'], convert_to_tensor=True).to(self.device)
                    current_embedding = self.dim_manager.align_embeddings(rule_embedding, "rule")
                    self.logger.debug(
                        f"Rule ID: {rule_id} Aligned rule embedding (from source_text) shape: {current_embedding.shape}")
                    if current_embedding.shape[-1] != self.dim_manager.target_dim:
                        self.logger.warning(
                            f"Rule {rule_id} embedding (from source_text) dimension {current_embedding.shape[-1]} does not match target {self.dim_manager.target_dim}. Skipping rule.")
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
                # Stack embeddings to create a tensor with shape [num_valid_rules, target_dim]
                self.rule_embeddings = torch.stack(rule_embeddings_list).to(self.device)
                self.logger.info(f"Rule embeddings tensor created with shape: {self.rule_embeddings.shape}")
            except Exception as e:
                self.logger.error(f"Error stacking rule embeddings: {e}")
                self.rule_embeddings = None
        else:
            self.rule_embeddings = None

        self.logger.info(
            f"Rule index built successfully with {len(self.rules)} rules. Final count of valid rules: {len(self.rule_ids)}")

    def build_graph(self):
        """
        Build the knowledge graph (DAG) from the loaded rules.
        """
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
        """
        Add edges between related rules based on keyword overlap.
        """
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
        """
        Enhanced rule validation with support for different rule types.
        """
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
                if "confidence" not in rule:
                    rule["confidence"] = self._calculate_rule_confidence(rule)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error in rule validation: {str(e)}")
            return False

    def _calculate_rule_confidence(self, rule: Dict) -> float:
        """
        Calculate confidence score for a rule based on available information.
        """
        base_confidence = 0.7
        if "source_text" in rule:
            base_confidence += 0.1
        if "entity_types" in rule:
            base_confidence += 0.1
        if "type" in rule:
            base_confidence += 0.1
        return min(1.0, base_confidence)

    def add_dynamic_rules(self, new_rules: List[Dict]) -> None:
        """
        Add new rules dynamically with improved validation and tracking.
        Ensures any rule embeddings are moved to the correct device.
        """
        if not new_rules:
            self.logger.info("No new rules to add.")
            return

        valid_rules = {}
        for i, rule in enumerate(new_rules, start=len(self.rules)):
            if self._validate_rule_structure(rule):
                rule_id = f"rule_{i}"
                rule['id'] = rule_id
                rule['version'] = GraphSymbolicReasoner._get_next_version(rule_id, self.rules)
                rule['added_timestamp'] = datetime.now().isoformat()
                rule['confidence'] = self._calculate_rule_confidence(rule)
                if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                    rule['embedding'] = rule['embedding'].to(self.device)
                valid_rules[rule_id] = rule
            else:
                self.logger.warning(f"Invalid rule structure: {rule}")

        if valid_rules:
            self.rules.update(valid_rules)
            self.build_rule_index()
            self.build_graph()
            self.logger.info(f"Added {len(valid_rules)} new rules. Total rules: {len(self.rules)}")
            try:
                self._track_rule_addition(list(valid_rules.values()))
            except Exception as e:
                self.logger.error(f"Error tracking rule addition (non-critical): {str(e)}")

    def _track_rule_addition(self, valid_rules: List[Dict]):
        """
        Track metrics related to the addition of dynamic rules for academic evaluation.
        """
        num_rules_added = len(valid_rules)
        total_rules_now = len(self.rules)
        avg_confidence = float(np.mean([rule.get('confidence', 0.0) for rule in valid_rules])) if valid_rules else 0.0
        self.logger.info(
            f"Added {num_rules_added} rules. Total rules: {total_rules_now}. "
            f"Average confidence: {avg_confidence:.3f}"
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
        Calculate cosine similarity with proper shape handling and alignment.
        """
        try:
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)
            if rule_embedding.dim() == 1:
                rule_embedding = rule_embedding.unsqueeze(0)
            if query_embedding.shape[1] != rule_embedding.shape[1]:
                # Attempt to align embeddings if one has 384 dimensions
                if query_embedding.shape[1] == 384:
                    query_embedding = self.dim_manager.validation_layer(query_embedding)
                if rule_embedding.shape[1] == 384:
                    rule_embedding = self.dim_manager.validation_layer(rule_embedding)
                if query_embedding.shape[1] != rule_embedding.shape[1]:
                    raise ValueError(
                        f"Dimension mismatch after alignment: query {query_embedding.shape}, rule {rule_embedding.shape}")
            with torch.no_grad():
                similarity = F.cosine_similarity(query_embedding, rule_embedding, dim=1)
                return similarity.item()
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def process_query(self, query: str) -> List[str]:
        """
        Process a query using symbolic reasoning with improved error handling and multi-hop support.
        Additionally, tracks reasoning chain metrics.
        """
        try:
            # Encode and align query embedding
            query_embedding_raw = self.embedder.encode(query, convert_to_tensor=True).to(self.device)
            logger.debug(f"Raw query embedding shape: {query_embedding_raw.shape}")

            if query_embedding_raw is None:
                raise ValueError("Failed to encode query")

            query_embedding = self.dim_manager.align_embeddings(query_embedding_raw, "query")
            logger.debug(f"Aligned query embedding shape: {query_embedding.shape}")

            if query_embedding.shape[-1] != self.dim_manager.target_dim:
                raise ValueError(
                    f"Query embedding dimension {query_embedding.shape[-1]} does not match target {self.dim_manager.target_dim}")

            # New multi-hop logic: retrieve a list of rule IDs representing each hop.
            multi_hop_paths = self._some_graph_traversal(query_embedding)
            all_hop_embeddings = []
            for rule_id in multi_hop_paths:
                # Each rule has an embedding; add new dimension for stacking.
                hop_emb = self.rules[rule_id]['embedding']
                all_hop_embeddings.append(hop_emb.unsqueeze(0))  # shape [1, embedding_dim]

            if all_hop_embeddings:
                # Stack along a new dimension to produce shape [1, hop_count, embedding_dim]
                multi_hop_emb = torch.cat(all_hop_embeddings, dim=0).unsqueeze(0)
            else:
                # Fallback: use query_embedding with dummy hop dimension => [1, 1, embedding_dim]
                multi_hop_emb = query_embedding.unsqueeze(0).unsqueeze(0)

            # Use the multi-hop embeddings to traverse the graph.
            responses = self.traverse_graph_from_multi_hop(multi_hop_emb)

            # --- Track reasoning chain metrics ---
            chain_info = {
                'steps': responses,
                'reasoning_path': responses
            }
            self.reasoning_metrics['chains'].append(chain_info)

            self._update_reasoning_metrics(responses)

            if not responses:
                logger.info("No symbolic match found.")
                return ["No symbolic match found."]

            logger.info(f"Found {len(responses)} symbolic responses.")
            return responses

        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return [f"Error processing query: {str(e)}"]

    def _some_graph_traversal(self, query_embedding: torch.Tensor) -> List[str]:
        """
        Dummy graph traversal to simulate multi-hop path retrieval.
        Replace this with your actual multi-hop traversal logic.
        Returns a list of rule IDs.
        """
        return self.rule_ids

    def traverse_graph_from_multi_hop(self, multi_hop_emb: torch.Tensor) -> List[str]:
        """
        Process multi-hop embeddings to generate responses.
        For now, this dummy implementation returns responses for each hop.
        """
        responses = []
        # Assume multi_hop_emb shape is [1, hop_count, embedding_dim]
        hop_count = multi_hop_emb.size(1)
        for i in range(hop_count):
            if i < len(self.rule_ids):
                rule_id = self.rule_ids[i]
                rule = self.rules.get(rule_id, {})
                responses.append(rule.get('response', ''))
        return responses

    def _update_reasoning_metrics(self, responses: List[str]):
        """
        Update reasoning metrics for basic academic analysis.
        """
        chain_length = len(responses) if responses else 1
        self.reasoning_metrics['path_lengths'].append(chain_length)
        confidences = [self._calculate_response_confidence(response) for response in responses]
        self.reasoning_metrics['match_confidences'].extend(confidences)
        for response in responses:
            rules_used = self._identify_rules_used(response)
            for rule_id in rules_used:
                self.reasoning_metrics.setdefault('rule_utilization', defaultdict(int))[rule_id] += 1

    def _calculate_response_confidence(self, response: str) -> float:
        """
        Calculate a confidence score for a response.
        """
        confidence_factors = []
        length_score = min(len(response.split()) / 50, 1.0)
        confidence_factors.append(length_score)
        doc = self.nlp(response)
        structure_score = len([ent for ent in doc.ents]) / max(1, len(response.split()))
        confidence_factors.append(structure_score)
        return float(np.mean(confidence_factors))

    def _identify_rules_used(self, response: str) -> Set[str]:
        """
        Identify which rules contributed to a response by comparing embeddings.
        """
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
