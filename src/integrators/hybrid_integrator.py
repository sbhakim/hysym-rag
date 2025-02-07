# src/integrators/hybrid_integrator.py

import time
import logging
import torch
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta

from src.knowledge_integrator import AlignmentLayer

logger = logging.getLogger(__name__)

class HybridIntegrator:
    """
    HySym-RAG's core integration system combining symbolic and neural reasoning.
    Includes caching and optional batching to control memory usage.
    """

    def __init__(self, symbolic_reasoner, neural_retriever, resource_manager, query_expander=None, cache_ttl=3600, batch_size=1):
        self.symbolic_reasoner = symbolic_reasoner
        self.neural_retriever = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        self.performance_stats = {
            'symbolic_hits': 0,
            'neural_hits': 0,
            'cache_hits': 0,
            'total_queries': 0,
            'hybrid_successes': 0
        }
        # Initialize alignment layer.
        self.alignment_layer = AlignmentLayer(sym_dim=384, neural_dim=384, target_dim=384)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("HybridIntegrator initialized successfully with alignment layer")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def ensure_device_consistency(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        return [self.ensure_device(tensor) for tensor in tensors]

    def _generate_cache_key(self, query: str) -> str:
        if self.query_expander:
            expanded = self.query_expander.expand_query(query)
            return f"{query}::{expanded}"
        return query

    def _get_valid_cache(self, cache_key: str) -> Optional[Tuple[List[str], str]]:
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self.cache_ttl):
                self.performance_stats['cache_hits'] += 1
                return entry['result'], entry['source']
            else:
                del self.cache[cache_key]
        return None

    def _update_cache(self, cache_key: str, result: Tuple[List[str], str]) -> None:
        if len(self.cache) > 1000:
            oldest_key = min(self.cache.items(), key=lambda x: x[1]['timestamp'])[0]
            del self.cache[oldest_key]
        self.cache[cache_key] = {
            'result': result[0],
            'source': result[1],
            'timestamp': datetime.now()
        }

    def process_query(self, query: str, context: str) -> Tuple[List[str], str]:
        self.performance_stats['total_queries'] += 1
        cache_key = self._generate_cache_key(query)
        cached_result = self._get_valid_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            # Encode query embeddings.
            symbolic_emb = self.symbolic_reasoner.encode(query)
            neural_emb = self.neural_retriever.encode(query)
            symbolic_emb, neural_emb = self.ensure_device_consistency(symbolic_emb, neural_emb)
            if symbolic_emb.dim() == 1:
                symbolic_emb = symbolic_emb.unsqueeze(0)
            if neural_emb.dim() == 1:
                neural_emb = neural_emb.unsqueeze(0)
            if symbolic_emb.size(1) != self.alignment_layer.sym_projection[0].in_features:
                logger.warning("Transforming symbolic embedding to match expected dimensions.")
                symbolic_emb = torch.nn.Linear(symbolic_emb.size(1), 384).to(self.device)(symbolic_emb)
            if neural_emb.size(1) != self.alignment_layer.neural_projection[0].in_features:
                logger.error(f"Neural embedding size mismatch: {neural_emb.size(1)}")
                raise ValueError(f"Neural embedding size mismatch: {neural_emb.size(1)} vs expected {self.alignment_layer.neural_projection[0].in_features}")

            try:
                aligned_emb, confidence, debug_info = self.alignment_layer(symbolic_emb, neural_emb)
                logger.info(f"Alignment confidence: {confidence:.3f}")
            except Exception as alignment_error:
                logger.error(f"Error in alignment layer: {str(alignment_error)}. Falling back to neural embeddings.")
                aligned_emb = neural_emb
                confidence = 0.0

            if confidence > 0.7:
                result = self._process_symbolic(query)
                if result:
                    self.performance_stats['hybrid_successes'] += 1
                    self._update_cache(cache_key, (result, "hybrid"))
                    return result, "hybrid"
                result = self._process_neural_optimized(query, context)
                self._update_cache(cache_key, (result, "neural"))
                return result, "neural"
            else:
                result = self._process_neural_optimized(query, context)
                self._update_cache(cache_key, (result, "neural"))
                return result, "neural"

        except RuntimeError as runtime_error:
            if "CUDA" in str(runtime_error):
                logger.error(f"CUDA error: {str(runtime_error)}. Resetting GPU.")
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                logger.warning("Falling back to CPU due to CUDA error.")
            raise
        except Exception as generic_error:
            logger.error(f"Unexpected error: {str(generic_error)}")
            result = self._process_symbolic(query)
            if result:
                return result, "symbolic"
            return self._process_neural_optimized(query, context), "neural"

    def _process_symbolic(self, query: str) -> Optional[List[str]]:
        try:
            symbolic_results = self.symbolic_reasoner.process_query(query)
            if symbolic_results and "No symbolic match found." not in symbolic_results:
                self.performance_stats['symbolic_hits'] += 1
                return symbolic_results
            return None
        except Exception as e:
            logger.error(f"Error in symbolic processing: {str(e)}")
            return None

    def _process_neural_optimized(self, query: str, context: str) -> List[str]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            start_time = time.time()
            with torch.no_grad():
                result = [self.neural_retriever.retrieve_answer(context, query)]
            processing_time = time.time() - start_time
            self.resource_manager.neural_perf_times.append(processing_time)
            self.performance_stats['neural_hits'] += 1
            return result
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_performance_metrics(self) -> Dict[str, float]:
        total = max(1, self.performance_stats['total_queries'])
        return {
            'symbolic_ratio': self.performance_stats['symbolic_hits'] / total,
            'neural_ratio': self.performance_stats['neural_hits'] / total,
            'cache_hit_ratio': self.performance_stats['cache_hits'] / total,
            'hybrid_success_ratio': self.performance_stats['hybrid_successes'] / total,
            'average_neural_time': (
                sum(self.resource_manager.neural_perf_times) /
                len(self.resource_manager.neural_perf_times)
                if self.resource_manager.neural_perf_times else 0
            )
        }
