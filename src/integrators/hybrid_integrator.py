# src/integrators/hybrid_integrator.py

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from collections import defaultdict

from src.knowledge_integrator import AlignmentLayer
from src.reasoners.rg_retriever import RuleGuidedRetriever
from src.utils.device_manager import DeviceManager


class HybridIntegrator:
    """
    Enhanced HybridIntegrator for academic evaluation of symbolic-neural integration.

    This component implements the core hybrid reasoning approach of HySym-RAG,
    combining symbolic and neural processing with careful attention to:
    - Resource efficiency through adaptive processing
    - Multi-hop reasoning quality
    - Integration fidelity measurement
    - Academic metrics collection
    """

    def __init__(self,
                 symbolic_reasoner,
                 neural_retriever,
                 resource_manager,
                 query_expander=None,
                 cache_ttl: int = 3600,
                 fusion_threshold: float = 0.6,
                 max_reasoning_steps: int = 5):
        """
        Initialize the enhanced hybrid integration system.

        Args:
            symbolic_reasoner: Graph-based symbolic reasoning component
            neural_retriever: Neural retrieval component
            resource_manager: System resource management
            query_expander: Optional query expansion component
            cache_ttl: Cache time-to-live in seconds
            fusion_threshold: Minimum confidence for fused answers
            max_reasoning_steps: Maximum reasoning steps for multi-hop queries
        """
        # Core components
        self.symbolic_reasoner = symbolic_reasoner
        self.neural = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander

        # Fusion parameters
        self.fusion_threshold = fusion_threshold
        self.max_reasoning_steps = max_reasoning_steps

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Determine device
        self.device = DeviceManager.get_device()

        # Initialize AlignmentLayer for advanced symbolic-neural embedding fusion
        self.alignment_layer = AlignmentLayer(
            sym_dim=384,  # Updated symbolic embedding dimension
            neural_dim=768,  # Neural embedding dimension
            target_dim=768  # Target integration dimension
        ).to(self.device)

        # Initialize RuleGuidedRetriever for context filtering
        self.rg_retriever = RuleGuidedRetriever()

        # Hybrid integration metrics
        self.integration_metrics = {
            'alignment_scores': [],
            'fusion_quality': [],
            'reasoning_steps': [],
            'resource_usage': defaultdict(list)
        }

        # Additional fusion metrics for academic analysis
        self.fusion_metrics = defaultdict(list)

        # Track reasoning chains for pattern analysis
        self.reasoning_chains = defaultdict(list)

        # Caching
        self.cache = {}
        self.cache_ttl = cache_ttl

    def process_query(self,
                      query: str,
                      context: str,
                      query_complexity: float = 0.5,
                      supporting_facts: Optional[List[Tuple[str, int]]] = None
                      ) -> Tuple[str, str]:
        """
        Process query using hybrid symbolic-neural reasoning with possible multi-hop.

        Args:
            query: Input query text
            context: Context information for neural retrieval
            query_complexity: Complexity score for the query
            supporting_facts: Optional supporting facts for rule-guided retrieval

        Returns:
            (answer, source) The fused or fallback answer, plus the source label.
        """
        try:
            # Check cache
            cache_key = self._generate_cache_key(query, context)
            cached_result = self._get_cache(cache_key)
            if cached_result:
                return cached_result

            # Symbolic reasoning (initial pass)
            symbolic_result = self._process_symbolic(query)

            # Determine if multi-hop logic is needed
            is_multi_hop = self._is_multi_hop_query(query)

            if is_multi_hop:
                result = self._handle_multi_hop_query(
                    query,
                    context,
                    symbolic_result,
                    supporting_facts
                )
            else:
                result = self._handle_single_hop_query(
                    query,
                    context,
                    symbolic_result,
                    supporting_facts
                )

            # Store in cache
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            self.logger.error(f"Error in hybrid processing: {str(e)}")
            return ("Error processing query.", "error")

    def _process_symbolic(self, query: str) -> List[str]:
        """
        Process query using symbolic reasoning only.
        """
        return self.symbolic_reasoner.process_query(query)

    def _is_multi_hop_query(self, query: str) -> bool:
        """
        Heuristic to detect multi-hop reasoning needs.
        """
        multi_hop_indicators = [" and ", " then ", " after ", " before "]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in multi_hop_indicators)

    def _handle_single_hop_query(self,
                                 query: str,
                                 context: str,
                                 symbolic_result: List[str],
                                 supporting_facts: Optional[List[Tuple[str, int]]] = None
                                 ) -> Tuple[str, str]:
        """
        Handle single-hop queries with advanced symbolic-neural fusion.
        """
        # Generate symbolic guidance
        symbolic_guidance = self._get_symbolic_guidance(symbolic_result, {'question': query})

        # Apply rule-guided retrieval
        if supporting_facts:
            filtered_context = self.rg_retriever.filter_context_by_rules(
                context,
                symbolic_guidance,
                supporting_facts=supporting_facts
            )
        else:
            filtered_context = self.rg_retriever.filter_context_by_rules(
                context,
                symbolic_guidance
            )

        # Get neural result
        neural_answer = self.neural.retrieve_answer(
            filtered_context,
            query,
            symbolic_guidance=symbolic_guidance,
            query_complexity=0.5  # Single-hop complexity
        )

        # Fuse symbolic and neural outputs
        fused_answer, confidence, debug_info = self._fuse_symbolic_neural(
            query,
            symbolic_result,
            neural_answer,
            query_complexity=0.5
        )

        return (fused_answer, "hybrid")

    def _handle_multi_hop_query(self,
                                query: str,
                                context: str,
                                symbolic_result: List[str],
                                supporting_facts: Optional[List[Tuple[str, int]]] = None
                                ) -> Tuple[str, str]:
        """
        Handle multi-hop queries with multi-step reasoning.
        """
        try:
            # Extract naive multi-hop chain
            reasoning_chain = self._extract_reasoning_chain(query, context)

            # Track intermediate results for academic analysis
            intermediate_results = []
            current_context = context

            # Process each hop in the chain
            for hop in reasoning_chain:
                # Symbolic guidance for this hop
                symbolic_guidance = self._get_symbolic_guidance(symbolic_result, hop)

                # Apply rule-guided retrieval for sub-query
                filtered_context = self.rg_retriever.filter_context_by_rules(
                    current_context,
                    symbolic_guidance,
                    query_complexity=0.7  # Slightly higher for multi-hop
                )

                # Neural retrieval for the sub-query
                hop_response = self.neural.retrieve_answer(
                    filtered_context,
                    hop['question'],
                    symbolic_guidance=symbolic_guidance,
                    query_complexity=0.7
                )

                intermediate_results.append(hop_response)
                current_context += " " + hop_response

            # Combine multi-hop partial results
            final_answer = self._combine_hop_results(intermediate_results, reasoning_chain)

            # Basic metric tracking
            self.integration_metrics['reasoning_steps'].append(len(reasoning_chain))

            # Fuse final multi-hop answer with symbolic outputs
            fused_answer_multi_hop, confidence_multi_hop, debug_info_multi_hop = self._fuse_symbolic_neural(
                query,
                symbolic_result,
                final_answer,
                query_complexity=0.7
            )
            return (fused_answer_multi_hop, "hybrid")

        except Exception as e:
            self.logger.error(f"Error in multi-hop processing: {str(e)}")
            return ("Error processing multi-hop query.", "error")

    def _extract_reasoning_chain(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        Simple multi-hop chain extraction by splitting on "and".
        """
        subqueries = [q.strip() for q in query.split("and") if q.strip()]
        return [{"question": subq} for subq in subqueries]

    def _combine_hop_results(self,
                             results: List[str],
                             reasoning_chain: List[Dict[str, str]]) -> str:
        """
        Combine results from each hop in multi-hop reasoning.
        """
        if not results:
            return ""

        if len(results) == 1:
            return results[0]

        combined = "Based on multiple reasoning steps:\n\n"
        for idx, (res, hop) in enumerate(zip(results, reasoning_chain)):
            combined += f"Step {idx + 1}: {hop['question']}\n"
            combined += f"Answer: {res}\n\n"
        combined += f"Final Answer: {results[-1]}"
        return combined

    def _fuse_symbolic_neural(self,
                              query: str,
                              symbolic_result: Union[List[str], str],
                              neural_answer: str,
                              query_complexity: float = 0.5
                              ) -> Tuple[str, float, Dict]:
        """
        Enhanced fusion with explicit reasoning tracking and adaptive weighting.
        Includes a fallback mechanism that returns the neural answer if fusion fails.

        Args:
            query: Original query
            symbolic_result: List of symbolic reasoning results
            neural_answer: Neural retriever response
            query_complexity: Query complexity score

        Returns:
            (fused_response, confidence, debug_info)
        """
        try:
            # Convert symbolic outputs to a single text string for embedding
            symbolic_text = " ".join(symbolic_result) if isinstance(symbolic_result, list) else symbolic_result

            # Attempt to retrieve embeddings from reasoner/neural modules
            try:
                symbolic_emb = self.symbolic_reasoner.get_embedding(symbolic_text)
                neural_emb = self.neural.get_embedding(neural_answer)
            except AttributeError:
                # Fallback to direct encoding if get_embedding not available
                symbolic_emb = self.alignment_layer.sym_adapter(
                    self._encode_text(symbolic_text)
                )
                neural_emb = self.alignment_layer.neural_adapter(
                    self._encode_text(neural_answer)
                )

            # Ensure both embeddings are on the same device
            symbolic_emb, neural_emb = DeviceManager.ensure_same_device(symbolic_emb, neural_emb, self.device)

            # Perform alignment with confidence scoring
            aligned_emb, confidence, debug_info = self.alignment_layer(
                symbolic_emb,
                neural_emb,
                rule_confidence=query_complexity  # reusing this parameter as a "rule_confidence"
            )

            # Generate a reasoned response with chain info
            fused_response = self._generate_reasoned_response(
                query,
                symbolic_result,
                neural_answer,
                aligned_emb,
                confidence
            )

            # Track fusion metrics
            self._update_fusion_metrics(confidence, query_complexity, debug_info)
            return fused_response, confidence, debug_info

        except Exception as e:
            self.logger.warning(f"Fusion failed: {str(e)}, falling back to neural response")
            # Return the neural answer with a mid-level confidence and a fallback indicator
            return neural_answer, 0.5, {"fallback": True, "error": str(e)}

    def _generate_reasoned_response(self,
                                    query: str,
                                    symbolic_result: Union[List[str], str],
                                    neural_answer: str,
                                    aligned_emb: torch.Tensor,
                                    confidence: float) -> str:
        """
        Generate a final response with explicit reasoning steps for interpretability.
        """
        # Convert symbolic result to list if needed
        symbolic_steps = symbolic_result if isinstance(symbolic_result, list) else [symbolic_result]

        # Initialize a chain to track each step
        reasoning_chain = []

        # Symbolic steps
        for step in symbolic_steps:
            if step and step.strip():
                reasoning_chain.append({
                    "type": "symbolic",
                    "content": step,
                    "confidence": confidence
                })

        # Neural step
        if neural_answer and neural_answer.strip():
            reasoning_chain.append({
                "type": "neural",
                "content": neural_answer,
                "confidence": confidence
            })

        # Save chain for future pattern analysis
        chain_id = hash(query)
        self.reasoning_chains[chain_id] = reasoning_chain

        # Decide how to combine
        if confidence >= self.fusion_threshold:
            # High confidence integrated answer
            response = "Based on integrated reasoning:\n\n"
            for idx, step in enumerate(reasoning_chain, 1):
                response += f"Step {idx} ({step['type']}): {step['content']}\n"
            response += f"\nFinal Answer (confidence: {confidence:.2f}): {neural_answer}"
        else:
            # Low confidence fallback
            response = f"Based on available information (confidence: {confidence:.2f}):\n{neural_answer}"

        return response

    def _update_fusion_metrics(self,
                               confidence: float,
                               query_complexity: float,
                               debug_info: Dict):
        """
        Track performance metrics for the fusion process.
        """
        self.fusion_metrics['confidence'].append(confidence)
        self.fusion_metrics['complexity'].append(query_complexity)
        if 'attention_weights' in debug_info:
            self.fusion_metrics['attention_patterns'].append(debug_info['attention_weights'])

    def get_fusion_analysis(self) -> Dict[str, Any]:
        """
        Return comprehensive analysis of fusion performance for academic studies.
        """
        if not self.fusion_metrics['confidence']:
            # No metrics collected yet
            return {}

        conf_values = self.fusion_metrics['confidence']
        comp_values = self.fusion_metrics['complexity']

        analysis = {
            'average_confidence': float(np.mean(conf_values)),
            'confidence_std': float(np.std(conf_values)),
            'fusion_count': len(conf_values),
            'complexity_correlation': 0.0,
            'reasoning_patterns': self._analyze_reasoning_patterns(),
            'fusion_success_rate': self._calculate_fusion_success_rate()
        }

        if len(conf_values) == len(comp_values) and len(conf_values) > 1:
            # Calculate correlation between confidence and complexity
            analysis['complexity_correlation'] = float(
                np.corrcoef(conf_values, comp_values)[0, 1]
            )

        return analysis

    def _analyze_reasoning_patterns(self) -> Dict[str, float]:
        """
        Analyze the distribution of reasoning chain patterns based on step types.
        """
        pattern_stats = defaultdict(int)
        total_chains = len(self.reasoning_chains)
        if total_chains == 0:
            return {}

        for chain in self.reasoning_chains.values():
            pattern = tuple(step['type'] for step in chain)
            pattern_stats[pattern] += 1

        return {
            str(pattern): count / total_chains
            for pattern, count in pattern_stats.items()
        }

    def _calculate_fusion_success_rate(self) -> float:
        """
        Calculate how often fused answers exceed the confidence threshold.
        """
        conf_list = self.fusion_metrics['confidence']
        if not conf_list:
            return 0.0
        successful_fusions = sum(1 for c in conf_list if c >= self.fusion_threshold)
        return successful_fusions / len(conf_list)

    def _process_multi_hop(self, query: str, context: str) -> Tuple[str, str]:
        """
        Legacy method for multi-hop: currently unused, placeholder for backward compatibility.
        """
        try:
            symbolic_result = self.symbolic_reasoner.process_query(query, context)
            neural_result = self.neural.retrieve_answer(context, query)
            fused_result = f"{symbolic_result} {neural_result}"
            return (fused_result, "hybrid")
        except Exception as e:
            self.logger.error(f"Error in multi-hop processing: {str(e)}")
            return ("Error processing query.", "error")

    def _get_symbolic_guidance(self,
                               symbolic_result: List[str],
                               hop: Dict[str, Any]) -> List[str]:
        """
        Generate symbolic guidance from the symbolic results for the current hop.
        """
        # Could be extended with more advanced logic
        return symbolic_result

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Helper to encode text into a tensor for fallback embedding usage.
        """
        from sentence_transformers import SentenceTransformer  # Late import if needed
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        return embedder.encode(text, convert_to_tensor=True).to(self.device)

    def _generate_cache_key(self, query: str, context: str) -> str:
        """
        Generate a unique cache key from query and context.
        """
        return f"{hash(query)}_{hash(context)}"

    def _get_cache(self, key: str) -> Optional[Tuple[str, str]]:
        """
        Retrieve a cached result if it hasn't expired.
        """
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            del self.cache[key]
        return None

    def _set_cache(self, key: str, result: Tuple[str, str]):
        """
        Store a result in cache with a timestamp.
        """
        self.cache[key] = (result, time.time())

    def get_integration_metrics(self) -> Dict[str, Any]:
        """
        Retrieve broader integration metrics for academic evaluation.
        """
        if not self.integration_metrics['reasoning_steps']:
            # No queries processed yet
            return {
                'alignment_quality': {'mean': 0.0, 'std': 0.0},
                'reasoning_depth': {'avg_steps': 0.0, 'max_steps': 0},
                'fusion_quality': {'mean': 0.0, 'std': 0.0}
            }

        return {
            'alignment_quality': {
                'mean': float(np.mean(self.integration_metrics['alignment_scores'])),
                'std': float(np.std(self.integration_metrics['alignment_scores']))
            },
            'reasoning_depth': {
                'avg_steps': float(np.mean(self.integration_metrics['reasoning_steps'])),
                'max_steps': max(self.integration_metrics['reasoning_steps'])
            },
            'fusion_quality': {
                'mean': float(np.mean(self.integration_metrics['fusion_quality'])),
                'std': float(np.std(self.integration_metrics['fusion_quality']))
            }
        }
