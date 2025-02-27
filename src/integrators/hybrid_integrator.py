# src/integrators/hybrid_integrator.py

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from collections import defaultdict
import math
from datetime import datetime

from src.knowledge_integrator import AlignmentLayer
from src.reasoners.rg_retriever import RuleGuidedRetriever
from src.utils.device_manager import DeviceManager
from src.utils.dimension_manager import DimensionalityManager
from src.reasoners.dummy_reasoners import DummySymbolicReasoner # Import DummySymbolicReasoner


logger = logging.getLogger(__name__)


class HybridIntegrator:
    """
    Enhanced HybridIntegrator for academic evaluation of symbolic-neural integration.
    """

    def __init__(self, symbolic_reasoner, neural_retriever, resource_manager,
                 query_expander=None, cache_ttl: int = 3600,
                 fusion_threshold: float = 0.6, max_reasoning_steps: int = 5,
                 dim_manager: Optional[DimensionalityManager] = None):
        self.symbolic_reasoner = symbolic_reasoner
        self.neural = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander
        self.fusion_threshold = fusion_threshold
        self.max_reasoning_steps = max_reasoning_steps
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG in HybridIntegrator
        self.device = DeviceManager.get_device()
        self.dim_manager = dim_manager or DimensionalityManager(target_dim=768, device=self.device)
        self.alignment_layer = AlignmentLayer(
            sym_dim=384,
            neural_dim=768,
            target_dim=768,
            dim_manager=self.dim_manager
        ).to(self.device)
        self.rg_retriever = RuleGuidedRetriever()
        self.integration_metrics = {
            'alignment_scores': [],
            'fusion_quality': [],
            'reasoning_steps': [],
            'step_success': [],
            'path_lengths': [],
            'resource_usage': defaultdict(list)
        }
        self.fusion_metrics = defaultdict(list)
        self.reasoning_chains = defaultdict(dict)
        self.cache = {}
        self.cache_ttl = cache_ttl

        # Track component execution times
        self.component_times = {
            "symbolic": 0.0,
            "neural": 0.0
        }

    def process_query(self, query: str, context: str, query_complexity: float = 0.5,
                      supporting_facts: Optional[List[Tuple[str, int]]] = None) -> Tuple[str, str]:
        try:
            cache_key = self._generate_cache_key(query, context)
            cached_result = self._get_cache(cache_key)
            if cached_result:
                return cached_result

            # Track component timing
            symbolic_start = time.time()
            symbolic_result = self._process_symbolic(query)
            symbolic_time = time.time() - symbolic_start
            self.component_times["symbolic"] = symbolic_time

            is_multi_hop = self._is_multi_hop_query(query)

            neural_start = time.time()
            if is_multi_hop:
                result = self._handle_multi_hop_query(query, context, symbolic_result, supporting_facts)
            else:
                result = self._handle_single_hop_query(query, context, symbolic_result, supporting_facts)
            neural_time = time.time() - neural_start
            self.component_times["neural"] = neural_time

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            self.logger.error(f"Error in hybrid processing: {str(e)}")
            return ("Error processing query.", "error")

    def _process_symbolic(self, query: str) -> List[str]:
        return self.symbolic_reasoner.process_query(query)

    def _is_multi_hop_query(self, query: str) -> bool:
        multi_hop_indicators = [" and ", " then ", " after ", " before "]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in multi_hop_indicators)

    def _handle_single_hop_query(self, query: str, context: str, symbolic_result: List[str],
                                 supporting_facts: Optional[List[Tuple[str, int]]] = None) -> Tuple[str, str]:
        symbolic_guidance = self._get_symbolic_guidance(symbolic_result, {'question': query})
        if supporting_facts:
            filtered_context = self.rg_retriever.filter_context_by_rules(context, symbolic_guidance,
                                                                         supporting_facts=supporting_facts)
        else:
            filtered_context = self.rg_retriever.filter_context_by_rules(context, symbolic_guidance)
        neural_answer = self.neural.retrieve_answer(filtered_context, query,
                                                    symbolic_guidance=symbolic_guidance,
                                                    query_complexity=0.5)
        fused_answer, confidence, debug_info = self._fuse_symbolic_neural(
            query, symbolic_result, neural_answer, query_complexity=0.5
        )
        return (fused_answer, "hybrid")

    def _handle_multi_hop_query(self, query: str, context: str, symbolic_result: List[str],
                                supporting_facts: Optional[List[Tuple[str, int]]] = None) -> Tuple[str, str]:
        """
        Enhanced multi-hop query handling with robust reasoning chain construction.
        """
        try:
            # Use query_expander's decomposition if available; else fallback to basic extraction.
            if self.query_expander and hasattr(self.query_expander, 'decompose_query'):
                reasoning_chain = self.query_expander.decompose_query(query)
            else:
                reasoning_chain = self._extract_reasoning_chain(query, context)

            if not reasoning_chain:
                self.logger.warning("Could not extract reasoning chain")
                return self._handle_single_hop_query(query, context, symbolic_result, supporting_facts)

            intermediate_results = []
            current_context = context
            accumulated_knowledge = ""

            # Process each hop in the reasoning chain
            for hop_idx, hop in enumerate(reasoning_chain):
                symbolic_guidance = self._get_symbolic_guidance(symbolic_result, hop)
                enriched_context = current_context
                if accumulated_knowledge:
                    enriched_context = f"{current_context}\n\nPrevious findings:\n{accumulated_knowledge}"
                filtered_context = self.rg_retriever.filter_context_by_rules(
                    enriched_context,
                    symbolic_guidance,
                    query_complexity=0.7
                )
                hop_response = self.neural.retrieve_answer(
                    filtered_context,
                    hop['question'],
                    symbolic_guidance=symbolic_guidance,
                    query_complexity=0.7
                )
                if hop_response and not hop_response.startswith("Error"):
                    intermediate_results.append({
                        'hop_idx': hop_idx,
                        'question': hop['question'],
                        'response': hop_response,
                        'context_used': filtered_context
                    })
                    accumulated_knowledge += f"\nStep {hop_idx + 1}: {hop_response}"
                    current_context += " " + hop_response

            # If no intermediate results, fallback to single-hop processing
            if not intermediate_results:
                self.logger.warning(f"No intermediate results for query: {query}")
                return self._handle_single_hop_query(query, context, symbolic_result, supporting_facts)

            # Create a structured representation of the reasoning chain
            reasoning_chain_info = {
                'chain_id': hash(query),
                'query': query,
                'chain_length': len(intermediate_results),
                'hop_count': len(reasoning_chain),
                'steps': [],
                'overall_confidence': 0.0
            }

            # Calculate confidence scores for each hop and track them
            hop_confidences = []
            for idx, (hop, res) in enumerate(zip(reasoning_chain, intermediate_results)):
                base_conf = 0.5 + (0.1 * idx)
                response_length = len(res.get('response', '').split())
                length_boost = min(0.2, response_length / 500)  # Cap at 0.2
                hop_conf = min(0.95, base_conf + length_boost)
                hop_confidences.append(hop_conf)
                reasoning_chain_info['steps'].append({
                    'hop_idx': idx,
                    'question': hop.get('question', ''),
                    'response': res.get('response', ''),
                    'confidence': hop_conf,
                    'type': 'multi_hop'
                })
            if hop_confidences:
                reasoning_chain_info['overall_confidence'] = sum(hop_confidences) / len(hop_confidences)

            # Store metrics in integration_metrics
            self.integration_metrics['reasoning_steps'].append(len(reasoning_chain))
            self.integration_metrics['path_lengths'].append(len(intermediate_results))

            # Initialize metrics collections if they don't exist
            if 'chain_lengths' not in self.integration_metrics:
                self.integration_metrics['chain_lengths'] = []
            if 'chain_confidences' not in self.integration_metrics:
                self.integration_metrics['chain_confidences'] = []
            if 'inference_depths' not in self.integration_metrics:
                self.integration_metrics['inference_depths'] = []

            # Track detailed reasoning metrics for academic evaluation
            self.integration_metrics['chain_lengths'].append(reasoning_chain_info['chain_length'])
            self.integration_metrics['chain_confidences'].append(reasoning_chain_info['overall_confidence'])
            self.integration_metrics['inference_depths'].append(reasoning_chain_info['hop_count'])

            # Also track reasoning metrics in symbolic reasoner for cross-component analysis
            if hasattr(self.symbolic_reasoner, 'reasoning_metrics'):
                self.symbolic_reasoner.reasoning_metrics['path_lengths'].append(reasoning_chain_info['chain_length'])
                self.symbolic_reasoner.reasoning_metrics['match_confidences'].append(
                    reasoning_chain_info['overall_confidence'])
                hop_count = reasoning_chain_info['hop_count']
                self.symbolic_reasoner.reasoning_metrics['hop_distributions'][hop_count] += 1
                self.symbolic_reasoner.reasoning_metrics['pattern_types']['multi_hop'] += 1

            # Store the reasoning chain
            self.reasoning_chains[reasoning_chain_info['chain_id']] = reasoning_chain_info

            final_answer = self._combine_hop_results(
                [res['response'] for res in intermediate_results],
                reasoning_chain
            )

            fused_answer, confidence, debug_info = self._fuse_symbolic_neural(
                query, symbolic_result, final_answer, query_complexity=0.7
            )

            debug_info['reasoning_chain'] = reasoning_chain_info

            return (fused_answer, "hybrid")

        except Exception as e:
            self.logger.error(f"Error in multi-hop processing: {str(e)}")
            return self._handle_single_hop_query(query, context, symbolic_result, supporting_facts)

    def _extract_reasoning_chain(self, query: str, context: str) -> List[Dict[str, str]]:
        # Basic fallback: decompose query by splitting on 'and'
        subqueries = [q.strip() for q in query.split("and") if q.strip()]
        return [{"question": subq} for subq in subqueries]

    def _combine_hop_results(self, results: List[str], reasoning_chain: List[Dict[str, str]]) -> str:
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

    def _assess_symbolic_contribution(self, symbolic_result: Union[List[str], str]) -> float:
        """
        Assess the quality/quantity of symbolic matches.
        Returns a contribution score capped at 0.3.
        """
        if isinstance(symbolic_result, list) and symbolic_result:
            if symbolic_result[0].strip().lower().startswith("no symbolic match"):
                return 0.0
            num_matches = len(symbolic_result)
            return min(0.3, num_matches * 0.1)
        return 0.0

    def _fuse_symbolic_neural(self, query: str, symbolic_result: Union[List[str], str], neural_answer: str,
                              query_complexity: float = 0.5) -> Tuple[str, float, Dict]:
        try:
            # Format symbolic results consistently
            symbolic_text = ""
            confidence_scores = []

            # Handle different symbolic result formats
            if isinstance(symbolic_result, dict) and "response" in symbolic_result:
                symbolic_responses = symbolic_result["response"][:3] if isinstance(symbolic_result["response"],
                                                                                   list) else [
                    symbolic_result["response"]]
                confidence_scores = symbolic_result.get("similarities", [0.7] * len(symbolic_responses))
                symbolic_text = "\n".join(symbolic_responses)
            elif isinstance(symbolic_result, list):
                symbolic_responses = symbolic_result[:3]  # Limit to top 3 responses
                confidence_scores = [0.7] * len(symbolic_responses)  # Default confidence
                symbolic_text = "\n".join(symbolic_responses)
            else:
                symbolic_text = str(symbolic_result)
                confidence_scores = [0.7]  # Default confidence

            # FIXED: Check if neural_answer is valid
            if not neural_answer or (
                    isinstance(neural_answer, str) and neural_answer.strip() == ""):  # Corrected condition
                self.logger.warning("Empty neural answer received, generating fallback response")
                neural_answer = "No additional information available from neural processing."

            # Ensure neural answer is a string
            if not isinstance(neural_answer, str):
                neural_answer = str(neural_answer)

            # Get embeddings with robust error handling
            try:
                symbolic_emb_raw = None  # Initialize to None
                # CHECK: Only get symbolic embedding if not using DummySymbolicReasoner
                if not isinstance(self.symbolic_reasoner, DummySymbolicReasoner):  # Conditional check here
                    symbolic_emb_raw = self._get_symbolic_embedding(symbolic_text)
                neural_emb_raw = self._get_neural_embedding(neural_answer)
            except Exception as embedding_error:
                self.logger.error(f"Error generating embeddings: {embedding_error}")
                # FALLBACK: Construct a basic fusion without embedding alignment
                fallback_response = self._create_fallback_fusion(query, symbolic_text, neural_answer)
                return fallback_response, 0.5, {"error": str(embedding_error), "fallback": "basic_fusion"}

            # Process through alignment
            try:
                symbolic_emb = None # Initialize to None
                if symbolic_emb_raw is not None: # Only align if we have a symbolic embedding
                    # Align dimensions safely
                    symbolic_emb = self.dim_manager.align_embeddings(symbolic_emb_raw, "symbolic")


                neural_emb_raw = self._get_neural_embedding(neural_answer) # Neural emb is still needed
                neural_emb = self.dim_manager.align_embeddings(neural_emb_raw, "neural")

                # Move to device
                neural_emb = DeviceManager.ensure_same_device(neural_emb, neural_emb, self.device)[0] # Only neural_emb needs device moved


                aligned_emb = neural_emb # Default to neural_emb if no symbolic emb for alignment
                confidence = 0.9 # Default high confidence if only neural
                debug_info = {}

                if symbolic_emb is not None: # Proceed with alignment layer only if symbolic_emb exists
                    # Move to device (both if using alignment)
                    symbolic_emb, neural_emb = DeviceManager.ensure_same_device(symbolic_emb, neural_emb, self.device)

                    # Process through alignment layer
                    aligned_emb, confidence, debug_info = self.alignment_layer(
                        symbolic_emb,
                        neural_emb,
                        rule_confidence=query_complexity
                    )


                # Generate the fused response with proper formatting
                fused_response = self._generate_reasoned_response(query, symbolic_result, neural_answer, confidence)

                # Update metrics
                self._update_fusion_metrics(confidence, query_complexity, debug_info)

                return fused_response, confidence, debug_info

            except Exception as alignment_error:
                self.logger.error(f"Alignment error: {alignment_error}")
                # FALLBACK: We'll at least show both answers with a proper explanation
                fallback_response = self._create_fallback_fusion(query, symbolic_text, neural_answer)
                return fallback_response, 0.5, {"error": str(alignment_error), "fallback": "basic_fusion"}

        except Exception as e:
            error_message = str(e)
            self.logger.warning(f"Fusion failed: {error_message}, falling back to neural response")
            return neural_answer, 0.5, {"fallback": True, "error": error_message}

    def _create_fallback_fusion(self, query: str, symbolic_text: str, neural_answer: str) -> str:
        """Create a simple fusion when advanced alignment fails"""
        # Simple fallback that doesn't rely on complex embedding or alignment
        if len(symbolic_text.strip()) > 10:
            response = f"Based on available information:\n\n"
            response += f"Background knowledge:\n{symbolic_text}\n\n"
            response += f"Analysis:\n{neural_answer}"
        else:
            response = neural_answer

        return response

    def _generate_reasoned_response(self, query: str, symbolic_result: Union[List[str], str], neural_answer: str,
                                    confidence: float) -> str:
        # Handle various symbolic result formats
        if isinstance(symbolic_result, list):
            symbolic_steps = symbolic_result
        elif isinstance(symbolic_result, dict):
            # Extract responses from dictionary
            if "response" in symbolic_result:
                responses = symbolic_result["response"]
                if isinstance(responses, list):
                    symbolic_steps = responses
                else:
                    symbolic_steps = [responses]
            else:
                # If no response field, convert dict to string
                symbolic_steps = [str(symbolic_result)]
        else:
            symbolic_steps = [str(symbolic_result)]

        # Create reasoning chain
        reasoning_chain = []
        for step in symbolic_steps:
            # Convert to string and check if non-empty
            step_str = str(step)
            if step_str and step_str.strip():
                reasoning_chain.append({"type": "symbolic", "content": step_str, "confidence": confidence})

        # Add neural answer if available
        if neural_answer and isinstance(neural_answer, str) and neural_answer.strip():
            reasoning_chain.append({"type": "neural", "content": neural_answer, "confidence": confidence})

        # Store the reasoning chain
        chain_id = hash(query)
        self.reasoning_chains[chain_id] = reasoning_chain

        # Format the response based on confidence
        if confidence >= self.fusion_threshold:
            response = "Based on integrated reasoning:\n\n"
            for idx, step in enumerate(reasoning_chain, 1):
                response += f"Step {idx} ({step['type']}): {step['content']}\n"
            response += f"\nFinal Answer (confidence: {confidence:.2f}): {neural_answer}"
        else:
            response = f"Based on available information (confidence: {confidence:.2f}):\n{neural_answer}"

        return response

    def _update_fusion_metrics(self, confidence: float, query_complexity: float, debug_info: Dict):
        self.fusion_metrics['confidence'].append(confidence)
        self.fusion_metrics['complexity'].append(query_complexity)
        if 'attention_weights' in debug_info:
            self.fusion_metrics['attention_patterns'].append(debug_info['attention_weights'])

    def _get_symbolic_embedding(self, symbolic_result: Union[List[str], str]) -> torch.Tensor:
        symbolic_text = " ".join(symbolic_result) if isinstance(symbolic_result, list) else symbolic_result
        return self.symbolic_reasoner.embedder.encode(symbolic_text, convert_to_tensor=True).to(self.device)

    def _get_neural_embedding(self, neural_answer: str) -> torch.Tensor:
        return self.neural.encoder.encode(neural_answer, convert_to_tensor=True).to(self.device)

    def get_integration_metrics(self) -> Dict[str, Any]:
        if not self.integration_metrics['reasoning_steps']:
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

    def _get_symbolic_guidance(self, symbolic_result: List[str], hop: Dict[str, Any]) -> List[str]:
        return symbolic_result

    def _answer_addresses_query(self, answer: str, query: str) -> bool:
        """Check if the answer actually addresses the query."""
        import re
        # Basic check - neural answer should contain key terms from the query
        query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        common_terms = query_terms.intersection(answer_terms)
        return len(common_terms) >= min(2, len(query_terms) / 2)

    def _estimate_answer_relevance(self, query: str, answer: str) -> float:
        """Estimate how relevant the answer is to the query."""
        import re
        query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))

        if not query_terms or not answer_terms:
            return 0.7  # Default value

        common_terms = query_terms.intersection(answer_terms)
        coverage = len(common_terms) / len(query_terms) if query_terms else 0

        # Check answer length - penalize very short answers
        length_factor = min(1.0, len(answer) / 100)

        return min(0.95, (0.6 + coverage * 0.4) * length_factor)

    def _encode_text(self, text: str) -> torch.Tensor:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        return embedder.encode(text, convert_to_tensor=True).to(self.device)

    def _generate_cache_key(self, query: str, context: str) -> str:
        return f"{hash(query)}_{hash(context)}"

    def _get_cache(self, key: str) -> Optional[Tuple[str, str]]:
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            del self.cache[key]
        return None

    def _set_cache(self, key: str, result: Tuple[str, str]):
        self.cache[key] = (result, time.time())