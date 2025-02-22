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
from src.utils.dimension_manager import DimensionalityManager  # Import DimensionalityManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphSymbolicReasoner:
    """
    Enhanced graph-based symbolic reasoner for academic evaluation of HySym-RAG.
    """

    def __init__(self,
                 rules_file: str,
                 match_threshold: float = 0.25,
                 max_hops: int = 5,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 device: Optional[torch.device] = None,
                 dim_manager: Optional[DimensionalityManager] = None  # Add DimensionalityManager
                 ):
        """
        Enhanced symbolic reasoner with proper tensor handling and academic metrics tracking.
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)


        self.match_threshold = match_threshold
        self.max_hops = max_hops
        self.device = device or DeviceManager.get_device()
        self.dim_manager = dim_manager # Store DimensionalityManager

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

        # Initialize academic tracking metrics
        self.reasoning_metrics = {
            'path_lengths': [],
            'match_confidences': [],
            'hop_distributions': defaultdict(int),
            'pattern_types': defaultdict(int)
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

        self.logger.info("GraphSymbolicReasoner initialized successfully")

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
                    rule['version'] = GraphSymbolicReasoner._get_next_version(rule_id, loaded_rules) # Corrected call
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
    def _get_next_version( rule_id: str, current_rules: Dict[str, Dict]) -> int:
        """
        Get the next version number for a rule.
        """
        return sum(1 for r in current_rules.values() if r.get('id', '').startswith(rule_id)) +1

    def build_rule_index(self):
        """
        Build an index for rules to speed up retrieval.
        """
        self.rule_index = {}
        self.rule_ids = []
        rule_embeddings_list = []

        for rule_id, rule in self.rules.items():
            self.rule_ids.append(rule_id)
            self.rule_index[rule_id] = rule

            if 'keywords' in rule:
                for keyword in rule['keywords']:
                    self.keyword_index[keyword].append(rule_id)

            if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                # Align rule embedding to target dimension during index building
                aligned_embedding = self.dim_manager.align_embeddings(rule['embedding'].to(self.device), "rule")
                rule['embedding'] = aligned_embedding # Replace with aligned embedding
                rule_embeddings_list.append(aligned_embedding) # Append aligned embedding
            elif 'source_text' in rule: # For rules dynamically extracted from HotpotQA that have 'source_text' but no pre-computed 'embedding'
                rule_embedding = self.embedder.encode(rule['source_text'], convert_to_tensor=True).to(self.device)
                aligned_embedding = self.dim_manager.align_embeddings(rule_embedding, "rule")
                rule['embedding'] = aligned_embedding
                rule_embeddings_list.append(aligned_embedding)


        if rule_embeddings_list:
            self.rule_embeddings = torch.stack(rule_embeddings_list).to(self.device)
            self.logger.info(f"Rule embeddings tensor created with shape: {self.rule_embeddings.shape}")
        else:
            self.rule_embeddings = None

        self.logger.info(f"Rule index built successfully with {len(self.rules)} rules.")


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
                has_required = all(field in rule for field in required_hotpot_fields)
                if has_required:
                    if "confidence" not in rule:
                        rule["confidence"] = self._calculate_rule_confidence(rule)
                    return True
                else:
                    return False
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
        for i, rule in enumerate(new_rules, start=len(self.rules)):  # Correct start index
            if self._validate_rule_structure(rule):
                rule_id = f"rule_{i}" #unique id
                rule['id'] = rule_id
                rule['version'] = GraphSymbolicReasoner._get_next_version(rule_id, self.rules) #corrected
                rule['added_timestamp'] = datetime.now().isoformat()
                rule['confidence'] = self._calculate_rule_confidence(rule)
                if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                     rule['embedding'] = rule['embedding'].to(self.device)
                valid_rules[rule_id] = rule # use a dictionary for correct usage in version tracking
            else:
                self.logger.warning(f"Invalid rule structure: {rule}")

        if valid_rules:
            self.rules.update(valid_rules) # use update for dictionary
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

    def process_query(self, query: str) -> List[str]:
        """
        Process a query using symbolic reasoning.
        """
        try:
            # Align query embedding to target dimension before processing
            query_embedding_raw = self.embedder.encode(query, convert_to_tensor=True).to(self.device)
            query_embedding = self.dim_manager.align_embeddings(query_embedding_raw, "symbolic")

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

    def _find_matching_rules(self, query_embedding: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Find matching rules with proper tensor comparison handling.
        """
        matches = []
        # Ensure query embedding is properly shaped
        query_embedding = query_embedding.view(1, -1)
        for rule_id, rule in self.rules.items():
            if 'embedding' in rule and isinstance(rule['embedding'], torch.Tensor):
                rule_embedding = rule['embedding'].to(self.device).view(1, -1)
                with torch.no_grad():
                    similarity = F.cosine_similarity(query_embedding, rule_embedding, dim=1)
                    sim_value = similarity.item()
                    if sim_value >= self.match_threshold:
                        matches.append((rule_id, sim_value))
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def traverse_graph(self, query_embedding: torch.Tensor) -> List[str]:
        """
        Traverse the knowledge graph to find relevant responses based on the query embedding.
        """
        responses = []
        matching_rules = self._find_matching_rules(query_embedding)
        for rule_id, sim_score in matching_rules:
            rule = self.rules.get(rule_id, {})
            responses.extend(self._process_rule(rule, {"subject": None}))
        return responses

    def _process_rule(self, rule: Dict, context: Dict[str, Any]) -> List[str]:
        """
        Process a rule to generate a response.
        """
        return [rule.get('response', '')]
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract relevant keywords from text using spaCy.
        """
        doc = self.nlp(text.lower())  # Use self.nlp here
        return [
            token.lemma_
            for token in doc
            if (token.pos_ in ('NOUN', 'VERB', 'ADJ') and
                not token.is_stop and
                len(token.text) > 2)
        ]
    def _update_reasoning_metrics(self, responses: List[str]):
        """
        Update reasoning metrics for academic analysis.
        """
        self.reasoning_metrics['path_lengths'].append(len(responses))
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
        doc = self.nlp(response)  # Use self.nlp for consistency
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

    # ------------------- New Methods for Enhanced Reasoning Chain Extraction -------------------

    def _extract_reasoning_chain(self, path: List[str], confidence_scores: List[float],
                                 query_id: Optional[str] = None) -> Dict:
        """
        Extract a structured reasoning chain from a given path through the graph.
        """
        reasoning_steps = []
        dependencies = nx.DiGraph()
        for idx, (node, base_conf) in enumerate(zip(path, confidence_scores)):
            rule = self.rules.get(node, {})
            prereqs = self._get_prerequisites(node)
            conclusions = self._get_conclusions(node)
            step_conf = self._calculate_step_confidence(rule, prereqs, base_conf)
            reasoning_steps.append({
                'step_id': idx,
                'rule': rule.get('response', ''),
                'confidence': step_conf,
                'prerequisites': prereqs,
                'conclusions': conclusions
            })
            if idx > 0:
                dependencies.add_edge(idx - 1, idx, weight=step_conf)
        chain_metrics = self._calculate_chain_metrics(reasoning_steps, dependencies)
        if query_id:
            self.reasoning_metrics.setdefault('chains', []).append({
                'query_id': query_id,
                'steps': reasoning_steps,
                'pattern': self.extract_reasoning_pattern("", path)
            })
            self.reasoning_metrics.setdefault('pattern_types', []).append(
                self.extract_reasoning_pattern("", path).get('pattern_type', 'unknown')
            )
        return {
            'steps': reasoning_steps,
            'overall_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
            'chain_length': len(path),
            'metrics': chain_metrics,
            'dependencies': self._serialize_dependencies(dependencies)
        }

    def extract_reasoning_pattern(self, query: str, path: List[str]) -> Dict[str, Any]:
        """
        Extract and analyze reasoning patterns for academic evaluation.
        """
        try:
            query_type = self._classify_query_type(query)
            pattern = self._analyze_path_pattern(path)
            steps = self._extract_intermediate_steps(path)
            return {
                'pattern_type': pattern.get('type', 'unknown'),
                'hop_count': len(path),
                'intermediate_facts': steps,
                'pattern_confidence': pattern.get('confidence', 0.0)
            }
        except Exception as e:
            self.logger.error(f"Error extracting reasoning pattern: {str(e)}")
            return {}

    def _classify_query_type(self, query: str) -> str:
        """
        Basic classification of query type.
        """
        if "compare" in query.lower() or "contrast" in query.lower():
            return "comparison"
        elif "?" in query:
            return "multi-hop"
        else:
            return "standard"

    def _analyze_path_pattern(self, path: List[str]) -> Dict[str, Any]:
        """
        Analyze the reasoning path pattern.
        """
        if len(path) <= 1:
            return {'type': 'linear', 'confidence': 1.0}
        unique_rules = set(path)
        if len(unique_rules) < len(path):
            return {'type': 'branching', 'confidence': 0.8}
        return {'type': 'linear', 'confidence': 0.9}

    def _extract_intermediate_steps(self, path: List[str]) -> List[str]:
        """
        Extract key intermediate facts from the reasoning path.
        """
        return [self.rules.get(node, {}).get('response', '') for node in path]

    def _calculate_chain_metrics(self, steps: List[Dict], dependencies: nx.DiGraph) -> Dict:
        """
        Calculate comprehensive chain metrics for academic analysis.
        """
        try:
            base_metrics = {
                'step_coherence': self._calculate_step_coherence(steps),
                'branching_factor': self._calculate_branching(dependencies),
                'path_linearity': self._calculate_linearity(dependencies)
            }
            academic_metrics = {
                'reasoning_depth': len(steps),
                'fact_utilization': self._calculate_fact_utilization(steps),
                'inference_quality': self._calculate_inference_quality(steps),
                'pattern_complexity': self._calculate_pattern_complexity(dependencies)
            }
            metrics = {**base_metrics, **academic_metrics}
            self._update_academic_metrics(metrics)
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating chain metrics: {str(e)}")
            return {}

    def _calculate_fact_utilization(self, steps: List[Dict]) -> float:
        """
        Calculate fact utilization as a placeholder metric.
        """
        utilized = sum(1 for step in steps if step.get('rule', '') != "")
        return utilized / len(steps) if steps else 0.0

    def _calculate_inference_quality(self, steps: List[Dict]) -> float:
        """
        Calculate inference quality as a placeholder (e.g., average confidence).
        """
        confidences = [step.get('confidence', 0.0) for step in steps]
        return float(np.mean(confidences)) if confidences else 0.0

    def _calculate_pattern_complexity(self, dependencies: nx.DiGraph) -> float:
        """
        Calculate pattern complexity based on the structure of the dependency graph.
        """
        if dependencies.number_of_nodes() == 0:
            return 0.0
        return float(dependencies.number_of_edges()) / dependencies.number_of_nodes()

    def _update_academic_metrics(self, metrics: Dict):
        """
        Update academic metrics tracking (placeholder implementation).
        """
        self.logger.info(f"Updated academic chain metrics: {metrics}")

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
                'mean_confidence': float(np.mean(self.reasoning_metrics['match_confidences'] or [0])),
                'confidence_distribution': np.histogram(self.reasoning_metrics['match_confidences'], bins=10)
            },
            'rule_utilization': dict(self.reasoning_metrics.get('rule_utilization', {})),
            'timing_analysis': {
                'mean_time': float(np.mean(self.reasoning_metrics.get('reasoning_times', [0]))),
                'std_time': float(np.std(self.reasoning_metrics.get('reasoning_times', [0])))
            },
            'rule_additions': self.reasoning_metrics.get('rule_additions', [])
        }

    def _calculate_path_distribution(self) -> Dict[int, float]:
        """
        Calculate distribution of reasoning path lengths.
        """
        path_counts = defaultdict(int)
        for length in self.reasoning_metrics['path_lengths']:
            path_counts[length] += 1
        total = len(self.reasoning_metrics['path_lengths'])
        return {length: count / total for length, count in path_counts.items()} if total else {}

    def _analyze_pattern_distribution(self) -> Dict[str, float]:
        """
        Analyze distribution of reasoning pattern types.
        """
        counts = defaultdict(int)
        total = len(self.reasoning_metrics.get('pattern_types', []))
        for p in self.reasoning_metrics.get('pattern_types', []):
            counts[p] += 1
        if total == 0:
            return {}
        return {ptype: counts[ptype] / total for ptype in counts}

    def _calculate_complexity_correlation(self) -> float:
        """
        Placeholder: Calculate correlation between chain length and average confidence.
        """
        lengths = self.reasoning_metrics['path_lengths']
        confidences = self.reasoning_metrics['match_confidences']
        if lengths and confidences and len(lengths) == len(confidences):
            return float(np.corrcoef(lengths, confidences)[0, 1])
        return 0.0

    def _calculate_pattern_success_rates(self) -> Dict[str, float]:
        """
        Placeholder: Calculate success rates for different reasoning patterns.
        """
        success_counts = defaultdict(int)
        total_counts = defaultdict(int)
        for chain in self.reasoning_metrics.get('chains', []):
            ptype = chain.get('pattern', {}).get('type', 'unknown')
            total_counts[ptype] += 1
            if chain.get('success', 0):
                success_counts[ptype] += 1

        return {
            ptype: (success_counts[ptype] / total_counts[ptype]) if total_counts[ptype] else 0.0
            for ptype in total_counts
        }

    def _get_complexity_metrics(self) -> Dict[str, float]:
        """
        Placeholder: Return additional complexity metrics.
        """
        return {
            'avg_chain_length': float(np.mean(self.reasoning_metrics['path_lengths'])),
            'std_chain_length': float(np.std(self.reasoning_metrics['path_lengths']))
        }

    def _analyze_pattern_effectiveness(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze effectiveness of different reasoning patterns.
        """
        effectiveness = defaultdict(list)
        for chain, ptype in zip(self.reasoning_metrics.get('chains', []),
                                self.reasoning_metrics.get('pattern_types', [])):
            effectiveness[ptype].append({
                'success': chain.get('success', 0),
                'confidence': chain.get('overall_confidence', 0.0)
            })
        result = {}
        for pattern_type, measures in effectiveness.items():
            success_rate = np.mean([m['success'] for m in measures])
            avg_confidence = np.mean([m['confidence'] for m in measures])
            result[pattern_type] = {
                'success_rate': float(success_rate),
                'avg_confidence': float(avg_confidence),
                'sample_size': len(measures)
            }
        return result

    def _get_ablation_metrics(self) -> Dict[str, Any]:
        """
        Placeholder: Return ablation study results.
        """
        return {}

    def _serialize_dependencies(self, dependencies: nx.DiGraph) -> Dict:
        """
        Serialize the dependencies graph into a dictionary format.
        """
        serialized = {}
        for u, v, data in dependencies.edges(data=True):
            serialized.setdefault(u, []).append({
                'to': v,
                'weight': data.get('weight', 0.0)
            })
        return serialized

    # ------------------- Utility / Placeholder methods for chain calculations -------------------

    def _get_prerequisites(self, node_id: str) -> List[str]:
        """
        Placeholder: Return prerequisite nodes for a rule.
        """
        return []

    def _get_conclusions(self, node_id: str) -> List[str]:
        """
        Placeholder: Return conclusion nodes for a rule.
        """
        return []

    def _calculate_step_confidence(self, rule: Dict, prereqs: List[str], base_conf: float) -> float:
        """
        Placeholder for more advanced step-by-step confidence calculation.
        """
        return base_conf

    def _calculate_step_coherence(self, steps: List[Dict]) -> float:
        """
        Placeholder: Evaluate how coherent each step is with the others.
        """
        return 1.0

    def _calculate_branching(self, dependencies: nx.DiGraph) -> float:
        """
        Placeholder: Evaluate branching factor of the dependency graph.
        """
        if dependencies.number_of_nodes() <= 1:
            return 0.0
        return float(dependencies.number_of_edges()) / dependencies.number_of_nodes()

    def _calculate_linearity(self, dependencies: nx.DiGraph) -> float:
        """
        Placeholder: Evaluate how linear the path is.
        """
        if dependencies.number_of_nodes() <= 1:
            return 1.0
        edges = dependencies.number_of_edges()
        nodes = dependencies.number_of_nodes()
        linear_ratio = float(edges) / (nodes - 1) if nodes > 1 else 1.0
        return min(1.0, linear_ratio)

    def _generate_cache_key(self, query: str, context: str) -> str:
        """
        Generate a unique cache key from query and context.
        """
        return f"{hash(query)}_{hash(context)}"

    def _get_cache(self, key: str) -> Optional[Tuple[str, str]]:
        """
        Retrieve a cached result if it hasn't expired.
        """
        if hasattr(self, 'cache') and key in self.cache:
            result, timestamp = self.cache[key]
            if (datetime.now().timestamp() - timestamp) < 3600:
                return result
            del self.cache[key]
        return None

    def _set_cache(self, key: str, result: Tuple[str, str]):
        """
        Store a result in cache with a timestamp.
        """
        if not hasattr(self, 'cache'):
            self.cache = {}
        self.cache[key] = (result, datetime.now().timestamp())