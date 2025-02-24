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
            'resource_usage': defaultdict(list)
        }
        self.fusion_metrics = defaultdict(list)
        self.reasoning_chains = defaultdict(list)
        self.cache = {}
        self.cache_ttl = cache_ttl

    def process_query(self, query: str, context: str, query_complexity: float = 0.5,
                      supporting_facts: Optional[List[Tuple[str, int]]] = None) -> Tuple[str, str]:
        try:
            cache_key = self._generate_cache_key(query, context)
            cached_result = self._get_cache(cache_key)
            if cached_result:
                return cached_result
            symbolic_result = self._process_symbolic(query)
            is_multi_hop = self._is_multi_hop_query(query)
            if is_multi_hop:
                result = self._handle_multi_hop_query(query, context, symbolic_result, supporting_facts)
            else:
                result = self._handle_single_hop_query(query, context, symbolic_result, supporting_facts)
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

            for hop_idx, hop in enumerate(reasoning_chain):
                # Get symbolic guidance for this hop
                symbolic_guidance = self._get_symbolic_guidance(symbolic_result, hop)
                # Update context with accumulated knowledge
                enriched_context = current_context
                if accumulated_knowledge:
                    enriched_context = f"{current_context}\n\nPrevious findings:\n{accumulated_knowledge}"
                # Filter context for this hop
                filtered_context = self.rg_retriever.filter_context_by_rules(
                    enriched_context,
                    symbolic_guidance,
                    query_complexity=0.7
                )
                # Get hop response using both symbolic and neural components
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
                    # Optionally update current_context for subsequent hops
                    current_context += " " + hop_response

            # Combine hop responses into a final answer
            final_answer = self._combine_hop_results(
                [res['response'] for res in intermediate_results],
                reasoning_chain
            )

            # Update integration metrics
            self.integration_metrics['reasoning_steps'].append(len(reasoning_chain))
            if len(reasoning_chain) > 0:
                self.integration_metrics['step_success'].append(len(intermediate_results) / len(reasoning_chain))
            else:
                self.integration_metrics['step_success'].append(0)

            # Final fusion with symbolic knowledge
            fused_answer, confidence, debug_info = self._fuse_symbolic_neural(
                query, symbolic_result, final_answer, query_complexity=0.7
            )
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

    def _fuse_symbolic_neural(self, query: str, symbolic_result: Union[List[str], str], neural_answer: str,
                              query_complexity: float = 0.5) -> Tuple[str, float, Dict]:
        try:
            symbolic_text = " ".join(symbolic_result) if isinstance(symbolic_result, list) else symbolic_result
            symbolic_emb_raw = self._get_symbolic_embedding(symbolic_result)
            neural_emb_raw = self._get_neural_embedding(neural_answer)

            self.logger.debug(f"Symbolic emb shape (before alignment): {symbolic_emb_raw.shape}")
            self.logger.debug(f"Neural emb shape (before alignment): {neural_emb_raw.shape}")

            symbolic_emb = self.dim_manager.align_embeddings(symbolic_emb_raw, "symbolic")
            neural_emb = self.dim_manager.align_embeddings(neural_emb_raw, "neural")

            self.logger.debug(f"Symbolic emb shape (after alignment): {symbolic_emb.shape}")
            self.logger.debug(f"Neural emb shape (after alignment): {neural_emb.shape}")

            symbolic_emb, neural_emb = DeviceManager.ensure_same_device(symbolic_emb, neural_emb, self.device)

            # Call the alignment layer to fuse embeddings.
            aligned_emb, confidence, debug_info = self.alignment_layer(
                symbolic_emb,
                neural_emb,
                rule_confidence=query_complexity
            )
            # Apply minimum confidence threshold
            MIN_CONFIDENCE = 0.1
            if confidence < MIN_CONFIDENCE:
                confidence = MIN_CONFIDENCE

            fused_response = self._generate_reasoned_response(query, symbolic_result, neural_answer, aligned_emb,
                                                              confidence)
            self._update_fusion_metrics(confidence, query_complexity, debug_info)
            return fused_response, confidence, debug_info

        except Exception as e:
            error_message = str(e)
            self.logger.warning(f"Fusion failed: {error_message}, falling back to neural response")
            return neural_answer, 0.5, {"fallback": True, "error": error_message}

    def _generate_reasoned_response(self, query: str, symbolic_result: Union[List[str], str], neural_answer: str,
                                    aligned_emb: torch.Tensor, confidence: float) -> str:
        symbolic_steps = symbolic_result if isinstance(symbolic_result, list) else [symbolic_result]
        reasoning_chain = []
        for step in symbolic_steps:
            if step and step.strip():
                reasoning_chain.append({"type": "symbolic", "content": step, "confidence": confidence})
        if neural_answer and neural_answer.strip():
            reasoning_chain.append({"type": "neural", "content": neural_answer, "confidence": confidence})
        chain_id = hash(query)
        self.reasoning_chains[chain_id] = reasoning_chain
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
