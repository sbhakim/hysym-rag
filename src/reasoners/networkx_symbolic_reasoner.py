# src/reasoners/networkx_symbolic_reasoner.py

import json
import spacy
import networkx as nx
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

class GraphSymbolicReasoner:
    """
    Enhanced graph-based symbolic reasoner for academic evaluation of HySym-RAG.

    Implements sophisticated graph-based reasoning with:
    - Multi-hop path analysis
    - Confidence scoring
    - Academic metric tracking
    - Dynamic rule integration and versioning
    """

    def __init__(self,
                 rules_file: str,
                 match_threshold: float = 0.25,
                 max_hops: int = 5,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize enhanced symbolic reasoner with academic tracking.

        Args:
            rules_file: Path to initial rules file.
            match_threshold: Threshold for rule matching.
            max_hops: Maximum reasoning hops.
            embedding_model: Model for semantic matching.
        """
        # Set core parameters
        self.match_threshold = match_threshold
        self.max_hops = max_hops

        # Initialize logger FIRST to ensure proper error tracking
        self.logger = logging.getLogger("GraphSymbolicReasoner")
        self.logger.setLevel(logging.INFO)

        # Initialize rules attribute and load rules from file.
        self.rules: List[Dict] = []
        loaded_rules = self.load_rules(rules_file)
        self.rules.extend(loaded_rules)

        # Initialize NLP components with error handling
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.embedder = SentenceTransformer(embedding_model)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Ensure the embedder is on the correct device
            self.embedder = self.embedder.to(self.device)
        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {str(e)}")
            raise

        # Initialize academic tracking metrics
        self.reasoning_metrics = {
            'path_lengths': [],
            'confidence_scores': [],
            'rule_utilization': defaultdict(int),
            'reasoning_times': [],
            'rule_additions': []  # Added to track dynamic rule additions
        }

        # Initialize knowledge graph and related indexes
        self.graph = nx.DiGraph()  # The knowledge graph (DAG)
        self.rule_index: Dict[str, Dict] = {}  # Index rules by rule_id
        self.rule_ids: List[str] = []  # List of rule_ids
        self.keyword_index = defaultdict(list)  # Inverted index for keywords
        self.entity_index = defaultdict(set)
        self.relation_index = defaultdict(set)
        self.rule_embeddings = None

        # Build indexes and graph
        self.build_rule_index()
        self.build_graph()  # Build the graph immediately

        self.logger.info("GraphSymbolicReasoner initialized successfully")

    def load_rules(self, rules_file: str) -> List[Dict]:
        """
        Load rules from a JSON file with enhanced validation and versioning.

        Args:
            rules_file: Path to the JSON rules file.

        Returns:
            List of rule dictionaries.
        """
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                loaded_rules = json.load(f)
                self.logger.info(f"Successfully loaded {len(loaded_rules)} rules from {rules_file}")
                # Validate and add version for each rule
                valid_rules = []
                for rule in loaded_rules:
                    if self._validate_rule_structure(rule):
                        rule['version'] = self._get_next_version(rule.get('id', 'default'))
                        valid_rules.append(rule)
                    else:
                        self.logger.warning(f"Invalid rule structure skipped: {rule}")
                return valid_rules
        except FileNotFoundError:
            self.logger.warning(f"Rules file not found at {rules_file}. Starting with empty rule set.")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON from {rules_file}. Please ensure it's valid JSON.")
            return []
        except Exception as e:
            self.logger.error(f"Error loading rules from {rules_file}: {str(e)}")
            return []

    def _get_next_version(self, rule_id: str) -> int:
        """
        Get the next version number for a rule.
        """
        # Count how many rules with the same id exist and add one.
        return sum(1 for rule in self.rules if rule.get('id', '') == rule_id) + 1

    def build_rule_index(self):
        """
        Build an index for rules to speed up retrieval.
        """
        self.rule_index = {}  # Reset index
        self.rule_ids = []
        self.rule_embeddings = []

        for i, rule in enumerate(self.rules):
            rule_id = f"rule_{i}"
            self.rule_ids.append(rule_id)
            self.rule_index[rule_id] = rule  # Index the rule by rule_id

            # Build keyword-based inverted index
            if 'keywords' in rule:
                for keyword in rule['keywords']:
                    self.keyword_index[keyword].append(rule_id)

            # If rule already has an embedding (as a tensor), store it
            if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                self.rule_embeddings.append(rule['embedding'])

        if self.rule_embeddings:
            self.rule_embeddings = torch.stack(self.rule_embeddings).to(self.device)
            self.logger.info(f"Rule embeddings tensor created with shape: {self.rule_embeddings.shape}")

        self.logger.info(f"Rule index built successfully with {len(self.rules)} rules.")

    def build_graph(self):
        """
        Build the knowledge graph (DAG) from the loaded rules.
        """
        try:
            self.graph = nx.DiGraph()  # Initialize directed graph
            # Add nodes with comprehensive attributes for each rule
            for rule_id, rule in self.rule_index.items():
                self.logger.debug(f"Debugging build_graph: rule_id={rule_id}, type(rule)={type(rule)}, rule={rule}")
                self.graph.add_node(
                    rule_id,
                    type="fact",
                    rule_response=rule.get('response', ''),
                    confidence=rule.get('confidence', 0.5),
                    version=rule.get('version', 0),
                    keywords=rule.get('keywords', [])
                )
                # Add edges based on keyword overlap
                self._add_rule_relationships(rule_id, rule)
            self.logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes")
        except Exception as e:
            self.logger.error(f"Error building knowledge graph: {str(e)}")
            self.graph = nx.DiGraph()  # Fallback to empty graph

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

    def add_dynamic_rules(self, new_rules: List[Dict]) -> None:
        """
        Add new rules dynamically with improved validation and tracking.
        """
        if not new_rules:
            self.logger.info("No new rules to add.")
            return

        valid_rules = []
        for rule in new_rules:
            if self._validate_rule_structure(rule):
                # Add metadata for dynamic rule
                rule['id'] = f"rule_{len(self.rules)}"
                rule['version'] = self._get_next_version(rule['id'])
                rule['added_timestamp'] = datetime.now().isoformat()
                rule['confidence'] = self._calculate_rule_confidence(rule)
                valid_rules.append(rule)
            else:
                self.logger.warning(f"Invalid rule structure: {rule}")

        if valid_rules:
            self.rules.extend(valid_rules)
            self.build_rule_index()
            self.build_graph()
            self.logger.info(f"Added {len(valid_rules)} new rules. Total rules: {len(self.rules)}")
            try:
                self._track_rule_addition(valid_rules)
            except Exception as e:
                self.logger.error(f"Error tracking rule addition (non-critical): {str(e)}")

    def _track_rule_addition(self, valid_rules: List[Dict]):
        """
        Track metrics related to the addition of dynamic rules for academic evaluation.

        Args:
            valid_rules: List of validated rules being added to the system
        """
        try:
            num_rules_added = len(valid_rules)
            total_rules_now = len(self.rules)
            avg_confidence = float(np.mean([rule.get('confidence', 0.0) for rule in valid_rules])) if valid_rules else 0.0

            # Log the metrics
            self.logger.info(
                f"Added {num_rules_added} rules. Total rules: {total_rules_now}. "
                f"Average confidence: {avg_confidence:.3f}"
            )

            # Update academic metrics
            if 'rule_additions' in self.reasoning_metrics:
                for rule in valid_rules:
                    self.reasoning_metrics['rule_additions'].append({
                        'timestamp': datetime.now().isoformat(),
                        'confidence': rule.get('confidence', 0.0),
                        'type': rule.get('type', 'unknown')
                    })
        except Exception as e:
            self.logger.error(f"Error tracking rule addition: {str(e)}")

    def _validate_rule_structure(self, rule: Dict) -> bool:
        """
        Enhanced rule validation with better support for different rule types.
        For HotpotQA-style rules, accept if the rule contains 'supporting_fact' along with required fields.
        For traditional rules, require both 'keywords' and 'response'.
        """
        try:
            if not isinstance(rule, dict):
                return False

            # Case 1: HotpotQA-style rules
            if "supporting_fact" in rule:
                required_hotpot_fields = {"type", "source_text", "keywords"}
                has_required = all(field in rule for field in required_hotpot_fields)
                if has_required:
                    if "confidence" not in rule:
                        rule["confidence"] = self._calculate_rule_confidence(rule)
                    return True
                else:
                    return False

            # Case 2: Traditional rules
            required_fields = {"keywords", "response"}
            has_required = all(field in rule for field in required_fields)
            if has_required:
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
        base_confidence = 0.7  # Default confidence
        if "source_text" in rule:
            base_confidence += 0.1
        if "entity_types" in rule:
            base_confidence += 0.1
        if "type" in rule:
            base_confidence += 0.1
        return min(1.0, base_confidence)

    def process_query(self, query: str) -> List[str]:
        """
        Public method to process a query using symbolic reasoning.
        Encodes the query and traverses the knowledge graph to generate responses.

        Args:
            query: The input query string.

        Returns:
            List of response strings.
        """
        try:
            query_embedding = self.embedder.encode(query, convert_to_tensor=True).to(self.device)
            responses = self.traverse_graph(query_embedding)
            self._update_reasoning_metrics(responses)
            if not responses:
                self.logger.info("No symbolic match found.")
                return ["No symbolic match found."]
            self.logger.info(f"Found {len(responses)} symbolic responses.")
            return responses
        except Exception as e:
            self.logger.error(f"Error in process_query: {str(e)}")
            return [f"Error processing query: {str(e)}"]

    def get_reasoning_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive reasoning metrics for academic evaluation.
        """
        return {
            'path_analysis': {
                'average_length': float(np.mean(self.reasoning_metrics['path_lengths'])),
                'max_length': float(max(self.reasoning_metrics['path_lengths'] or [0])),
                'path_distribution': self._calculate_path_distribution()
            },
            'confidence_analysis': {
                'mean_confidence': float(np.mean(self.reasoning_metrics['confidence_scores'] or [0])),
                'confidence_distribution': np.histogram(self.reasoning_metrics['confidence_scores'], bins=10)
            },
            'rule_utilization': dict(self.reasoning_metrics['rule_utilization']),
            'timing_analysis': {
                'mean_time': float(np.mean(self.reasoning_metrics['reasoning_times'] or [0])),
                'std_time': float(np.std(self.reasoning_metrics['reasoning_times'] or [0]))
            },
            'rule_additions': self.reasoning_metrics['rule_additions']
        }

    def _calculate_path_distribution(self) -> Dict[int, float]:
        """
        Calculate distribution of reasoning path lengths.
        """
        path_counts = defaultdict(int)
        for length in self.reasoning_metrics['path_lengths']:
            path_counts[length] += 1
        total = len(self.reasoning_metrics['path_lengths'])
        return {length: count / total for length, count in path_counts.items()}

    def _update_reasoning_metrics(self, responses: List[str]):
        """
        Update reasoning metrics for academic analysis.
        """
        self.reasoning_metrics['path_lengths'].append(len(responses))
        confidence_scores = [self._calculate_response_confidence(response) for response in responses]
        self.reasoning_metrics['confidence_scores'].extend(confidence_scores)
        for response in responses:
            rules_used = self._identify_rules_used(response)
            for rule_id in rules_used:
                self.reasoning_metrics['rule_utilization'][rule_id] += 1

    def _calculate_response_confidence(self, response: str) -> float:
        """
        Calculate a confidence score for a response.
        """
        confidence_factors = []
        # Length-based confidence
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
        for rule_id, rule in self.rule_index.items():
            if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                similarity = util.cos_sim(response_embedding, rule['embedding'].to(self.device)).item()
                if similarity > self.match_threshold:
                    used_rules.add(rule_id)
        return used_rules

    def traverse_graph(self, query_embedding: torch.Tensor) -> List[str]:
        """
        Traverse the knowledge graph to find relevant responses based on the query embedding.
        """
        responses = []
        initial_rule_matches = self._find_initial_rule_matches(query_embedding)
        for rule_id, similarity_score in initial_rule_matches:
            rule = self.rule_index[rule_id]
            responses.extend(self._process_rule(rule, {"subject": None}))
        return responses

    def _find_initial_rule_matches(self, query_embedding: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Find initial rules that match the query embedding based on cosine similarity.
        """
        similarities = util.cos_sim(query_embedding, self.rule_embeddings).squeeze(0)
        rule_matches = []
        for index, similarity_score in enumerate(similarities):
            if similarity_score >= self.match_threshold:
                rule_id = self.rule_ids[index]
                rule_matches.append((rule_id, similarity_score.item()))
        rule_matches.sort(key=lambda x: x[1], reverse=True)
        return rule_matches

    def _process_rule(self, rule: Dict, context: Dict[str, Any]) -> List[str]:
        """
        Process a rule to generate a response. This is a placeholder for more complex logic.
        """
        # For now, simply return the response text from the rule.
        return [rule.get('response', '')]

    def _is_comparison_question(self, query: str) -> bool:
        """Basic heuristic to detect comparison questions."""
        return "compare" in query.lower() or "contrast" in query.lower()

    def _handle_comparison_query(self, query: str) -> List[str]:
        """Placeholder for handling comparison queries."""
        self.logger.info("Comparison query detected, but handling not yet implemented.")
        return ["Comparison query processing is not yet implemented."]

    def _is_multi_hop_question(self, query: str) -> bool:
        """Heuristic to detect multi-hop questions (very basic)."""
        return "?" in query and ("and" in query.lower() or "then" in query.lower())

    def _handle_multi_hop_query(self, query: str) -> List[str]:
        """Placeholder for handling multi-hop queries."""
        self.logger.info("Multi-hop query detected, handling with basic logic.")
        query_embedding = self.embedder.encode(query, convert_to_tensor=True).to(self.device)
        responses = self.traverse_graph(query_embedding)
        if not responses:
            return ["No multi-hop symbolic match found."]
        return responses

    def _get_related_terms(self, term: str) -> List[str]:
        """
        Get semantically related terms using embeddings.
        """
        cache_key = hash(term)
        if cache_key in getattr(self, 'embedding_cache', {}):
            return self.embedding_cache[cache_key]

        try:
            term_emb = self.embedder.encode(term, convert_to_tensor=True).to(self.device)
            related = []
            for category, terms in self.expansion_rules.items():
                terms_emb = self.embedder.encode(terms, convert_to_tensor=True).to(self.device)
                similarities = util.cos_sim(term_emb, terms_emb)
                for idx, sim in enumerate(similarities[0]):
                    if sim > self.hop_threshold:
                        related.append(terms[idx])
            if not hasattr(self, 'embedding_cache'):
                self.embedding_cache = {}
            self.embedding_cache[cache_key] = related
            return related
        except Exception as e:
            self.logger.error(f"Error getting related terms for {term}: {str(e)}")
            return []

    def _extract_comparison_pairs(self, doc) -> List[Tuple[str, str]]:
        """
        Extract entity pairs for comparison queries.
        """
        pairs = []
        entities = list(doc.ents)
        for i in range(len(entities) - 1):
            for j in range(i + 1, len(entities)):
                between_tokens = doc[entities[i].end:entities[j].start]
                if any(token.text.lower() in ['versus', 'vs', 'or', 'and'] for token in between_tokens):
                    pairs.append((entities[i].text, entities[j].text))
        return pairs
