# src/integrators/hybrid_integrator.py

import time
import logging
import torch
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
from sentence_transformers import util

# Now we can use util.cos_sim() for cosine similarity calculations between embeddings

from src.knowledge_integrator import AlignmentLayer

logger = logging.getLogger(__name__)

class HybridIntegrator:
    """
    HySym-RAG's core integration system that combines symbolic and neural reasoning.
    """
    def __init__(self, symbolic_reasoner, neural_retriever, resource_manager, query_expander=None, cache_ttl=3600, batch_size=4):
        self.symbolic_reasoner = symbolic_reasoner
        self.neural = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander  # Also accessible via .expander property
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

    @property
    def expander(self):
        return self.query_expander

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
        """
        Process a query using hybrid symbolic-neural reasoning with adaptive computation.

        Args:
            query: The input query string
            context: Additional context information

        Returns:
            Tuple of (result list, reasoning type used)
        """
        # Track query statistics for performance monitoring
        self.performance_stats['total_queries'] += 1

        # Check cache first to avoid redundant computation
        cache_key = self._generate_cache_key(query)
        cached_result = self._get_valid_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            # Get query complexity to guide processing path
            if self.query_expander:
                complexity = self.query_expander.get_query_complexity(query)
                logger.info(f"Query complexity score: {complexity:.4f}")
            else:
                complexity = 0.5  # Default moderate complexity

            # Determine initial processing strategy based on complexity
            if complexity < 0.4:
                # Simple queries go straight to symbolic reasoning
                symbolic_result = self._process_symbolic(query)
                if symbolic_result:
                    return symbolic_result, "symbolic"

            # For moderate to complex queries, use hybrid approach
            symbolic_emb = self.symbolic_reasoner.encode(query)
            neural_emb = self.neural.encode(query)

            # Ensure consistent device placement and shape
            symbolic_emb, neural_emb = self.ensure_device_consistency(symbolic_emb, neural_emb)
            symbolic_emb = self._prepare_embedding(symbolic_emb, 300, "symbolic")
            neural_emb = self._prepare_embedding(neural_emb, 384, "neural")

            # Attempt alignment of symbolic and neural representations
            try:
                aligned_emb, confidence, debug_info = self.alignment_layer(symbolic_emb, neural_emb)
                logger.info(f"Alignment confidence: {confidence:.3f}")

                # Log detailed debug information if available
                if debug_info:
                    logger.debug(f"Alignment debug info: {debug_info}")

            except Exception as alignment_error:
                logger.error(f"Alignment error: {str(alignment_error)}. Falling back to neural.")
                confidence = 0.0

            # Process based on alignment confidence and complexity
            if confidence > 0.6:  # Lowered threshold for more hybrid processing
                symbolic_result = self._process_symbolic(query)
                if symbolic_result:
                    # Try hybrid processing with symbolic guidance
                    try:
                        neural_result = self._process_neural_optimized(
                            query,
                            context,
                            symbolic_guidance=symbolic_result
                        )
                        combined_result = self._combine_results(symbolic_result, neural_result)
                        self.performance_stats['hybrid_successes'] += 1
                        self._update_cache(cache_key, (combined_result, "hybrid"))
                        return combined_result, "hybrid"
                    except Exception as hybrid_error:
                        logger.warning(f"Hybrid processing failed: {str(hybrid_error)}")
                        # Fall back to symbolic result if hybrid fails
                        return symbolic_result, "symbolic"

            # Default to neural processing for low confidence or failed hybrid
            result = self._process_neural_optimized(query, context)
            self._update_cache(cache_key, (result, "neural"))
            return result, "neural"

        except RuntimeError as runtime_error:
            self._handle_runtime_error(runtime_error)
            raise

        except Exception as generic_error:
            logger.error(f"Unexpected error: {str(generic_error)}")
            return self._handle_fallback(query, context)

    def _prepare_embedding(self, embedding: torch.Tensor, target_dim: int,
                           name: str) -> torch.Tensor:
        """Helper method to prepare embeddings for alignment."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        if embedding.size(1) != target_dim:
            logger.warning(f"Transforming {name} embedding from {embedding.size(1)} to {target_dim}")
            projection = torch.nn.Linear(embedding.size(1), target_dim).to(self.device)
            embedding = projection(embedding)

        return embedding

    def _combine_results(self, symbolic_results: List[str],
                         neural_result: List[str]) -> List[str]:
        """Intelligently combine symbolic and neural results."""
        combined = []
        # Add symbolic insights first
        combined.extend(symbolic_results)

        # Add neural response if it provides new information
        for neural_resp in neural_result:
            if not any(self._high_overlap(neural_resp, sym_resp)
                       for sym_resp in symbolic_results):
                combined.append(neural_resp)

        return combined

    def _high_overlap(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check if two text segments have high semantic overlap."""
        emb1 = self.neural.encode(text1)
        emb2 = self.neural.encode(text2)
        similarity = util.cos_sim(emb1, emb2).item()
        return similarity > threshold

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
            result = [self.neural.retrieve_answer(context, query)]
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
            'average_neural_time': (sum(self.resource_manager.neural_perf_times) / len(self.resource_manager.neural_perf_times)
                                    if self.resource_manager.neural_perf_times else 0)
        }
