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
    HySym-RAG's core integration system that combines symbolic and neural reasoning.
    """

    def __init__(self, symbolic_reasoner, neural_retriever, resource_manager, query_expander=None, cache_ttl=3600,
                 batch_size=4):
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
        self.alignment_layer = AlignmentLayer(sym_dim=300, neural_dim=384, target_dim=768)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("HybridIntegrator initialized successfully with alignment layer")

    def ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensures the given tensor is on the correct device.
        """
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def ensure_device_consistency(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        """
        Ensure all tensors are moved to the same device as the primary tensor.
        """
        return [self.ensure_device(tensor) for tensor in tensors]

    def _generate_cache_key(self, query: str) -> str:
        """
        Generate a unique cache key, incorporating query expansion if available.
        """
        if self.query_expander:
            expanded = self.query_expander.expand_query(query)
            return f"{query}::{expanded}"
        return query

    def _get_valid_cache(self, cache_key: str) -> Optional[Tuple[List[str], str]]:
        """
        Retrieve a valid cache entry if one exists and hasn't expired.
        """
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self.cache_ttl):
                self.performance_stats['cache_hits'] += 1
                return entry['result'], entry['source']
            else:
                # Clean up expired entry
                del self.cache[cache_key]
        return None

    def _update_cache(self, cache_key: str, result: Tuple[List[str], str]) -> None:
        """
        Update the cache with a new result, managing cache size.
        """
        if len(self.cache) > 1000:
            oldest_key = min(self.cache.items(), key=lambda x: x[1]['timestamp'])[0]
            del self.cache[oldest_key]

        self.cache[cache_key] = {
            'result': result[0],
            'source': result[1],
            'timestamp': datetime.now()
        }

    def process_query(self, query: str, context: str) -> Tuple[List[str], str]:
        """
        Processes a query using the hybrid reasoning approach.
        """
        self.performance_stats['total_queries'] += 1
        cache_key = self._generate_cache_key(query)
        cached_result = self._get_valid_cache(cache_key)

        if cached_result:
            return cached_result

        try:
            # Step 1: Encode query into embeddings and ensure device consistency
            symbolic_emb = self.symbolic_reasoner.encode(query)
            neural_emb = self.neural_retriever.encode(query)
            symbolic_emb, neural_emb = self.ensure_device_consistency(symbolic_emb, neural_emb)

            # Step 2: Inspect and validate embeddings
            logger.debug(f"Raw symbolic embedding shape: {symbolic_emb.shape}")
            logger.debug(f"Raw neural embedding shape: {neural_emb.shape}")

            # Ensure embeddings are 2D
            if symbolic_emb.dim() == 1:
                symbolic_emb = symbolic_emb.unsqueeze(0)
            if neural_emb.dim() == 1:
                neural_emb = neural_emb.unsqueeze(0)

            # Validate dimensions against AlignmentLayer input
            if symbolic_emb.size(1) != self.alignment_layer.sym_projection[0].in_features:
                logger.warning("Transforming symbolic embedding to match the expected dimensions.")
                symbolic_emb = torch.nn.Linear(symbolic_emb.size(1), 300).to(self.device)(symbolic_emb)

            if neural_emb.size(1) != self.alignment_layer.neural_projection[0].in_features:
                logger.error(f"Neural embedding size mismatch: {neural_emb.size(1)}")
                raise ValueError(
                    f"Neural embedding size mismatch: {neural_emb.size(1)} vs expected {self.alignment_layer.neural_projection[0].in_features}")

            # Step 3: Perform alignment
            try:
                aligned_emb, confidence, debug_info = self.alignment_layer(symbolic_emb, neural_emb)
                logger.info(f"Alignment confidence: {confidence:.3f}")
            except Exception as alignment_error:
                logger.error(f"Error in alignment layer: {str(alignment_error)}. Falling back to neural embeddings.")
                aligned_emb = neural_emb
                confidence = 0.0

            # Step 4: Select reasoning path based on alignment confidence
            if confidence > 0.7:  # High confidence threshold
                result = self._process_symbolic(query)
                if result:
                    self.performance_stats['hybrid_successes'] += 1
                    self._update_cache(cache_key, (result, "hybrid"))
                    return result, "hybrid"

                # Fallback to neural processing if symbolic fails
                result = self._process_neural_optimized(query, context)
                self._update_cache(cache_key, (result, "neural"))
                return result, "neural"
            else:  # Low confidence: Default to neural processing
                result = self._process_neural_optimized(query, context)
                self._update_cache(cache_key, (result, "neural"))
                return result, "neural"

        except RuntimeError as runtime_error:
            # Handle CUDA-specific errors
            if "CUDA" in str(runtime_error):
                logger.error(f"CUDA error encountered: {str(runtime_error)}. Resetting GPU.")
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
        """
        Process a query using symbolic reasoning with error handling.
        """
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
        """
        Process a query using neural retrieval with memory optimization.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            start_time = time.time()
            result = [self.neural_retriever.retrieve_answer(context, query)]
            processing_time = time.time() - start_time
            self.resource_manager.neural_perf_times.append(processing_time)
            self.performance_stats['neural_hits'] += 1
            return result
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for system evaluation.
        """
        total = max(1, self.performance_stats['total_queries'])
        return {
            'symbolic_ratio': self.performance_stats['symbolic_hits'] / total,
            'neural_ratio': self.performance_stats['neural_hits'] / total,
            'cache_hit_ratio': self.performance_stats['cache_hits'] / total,
            'hybrid_success_ratio': self.performance_stats['hybrid_successes'] / total,
            'average_neural_time': (
                sum(self.resource_manager.neural_perf_times) /
                len(self.resource_manager.neural_perf_times)
                if self.resource_manager.neural_perf_times
                else 0
            )
        }
