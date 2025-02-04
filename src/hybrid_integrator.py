# src/hybrid_integrator.py
import time
import logging
import torch
from typing import Tuple, List, Dict, Optional, Any
from datetime import datetime, timedelta

# Import our enhanced alignment layer that bridges symbolic and neural representations
from src.knowledge_integrator import AlignmentLayer

# Configure logging
logger = logging.getLogger(__name__)

class HybridIntegrator:
    """
    HySym-RAG's core integration system that combines symbolic and neural reasoning.
    This implementation focuses on the key aspects of hybrid reasoning:
    1. Dynamic alignment between symbolic and neural representations
    2. Confidence-based processing path selection
    3. Intelligent caching for efficiency
    """

    def __init__(
            self,
            symbolic_reasoner,
            neural_retriever,
            resource_manager,
            query_expander=None,
            cache_ttl: int = 3600,  # Cache entries live for 1 hour by default
            batch_size: int = 4
    ):
        """
        Initialize the hybrid integration system with its essential components.

        Args:
            symbolic_reasoner: Component for symbolic reasoning
            neural_retriever: Component for neural retrieval and processing
            resource_manager: Manages computational resources
            query_expander: Optional component for query expansion
            cache_ttl: Time-to-live for cache entries in seconds
            batch_size: Size for batch processing when applicable
        """
        # Core reasoning components
        self.symbolic_reasoner = symbolic_reasoner
        self.neural_retriever = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander

        # Cache configuration
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size

        # Performance tracking metrics
        self.performance_stats = {
            'symbolic_hits': 0,
            'neural_hits': 0,
            'cache_hits': 0,
            'total_queries': 0,
            'hybrid_successes': 0
        }

        # Initialize the alignment layer that bridges symbolic and neural representations
        self.alignment_layer = AlignmentLayer(sym_dim=300, neural_dim=768)

        logger.info("HybridIntegrator initialized successfully with alignment layer")

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
        # Basic cache size management - remove oldest entry if cache gets too large
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
        Process a query using HySym-RAG's hybrid approach.

        This method implements the core of HySym-RAG's functionality:
        1. Check cache for existing results
        2. Perform symbolic-neural alignment
        3. Use alignment confidence to guide processing path
        4. Fall back to individual reasoning methods if needed
        """
        self.performance_stats['total_queries'] += 1

        # First check cache for existing results
        cache_key = self._generate_cache_key(query)
        cached_result = self._get_valid_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            # Prepare device for tensor operations (Ensure both embeddings are on the same device)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Get embeddings from both reasoning systems
            symbolic_emb = self.symbolic_reasoner.encode(query).to(device)
            neural_emb = self.neural_retriever.encode(query).to(device)

            # Perform alignment and get confidence score
            aligned_emb, confidence, debug_info = self.alignment_layer(symbolic_emb, neural_emb)

            logger.info(f"Alignment confidence: {confidence:.3f}")

            # Use confidence score to determine processing path
            if confidence > 0.7:  # High confidence threshold
                # Try symbolic reasoning first
                result = self._process_symbolic(query)
                if result:
                    self.performance_stats['hybrid_successes'] += 1
                    self._update_cache(cache_key, (result, "hybrid"))
                    return result, "hybrid"

                # Fall back to neural if symbolic fails
                result = self._process_neural_optimized(query, context)
                self._update_cache(cache_key, (result, "neural"))
                return result, "neural"
            else:
                # Low confidence - use neural processing
                result = self._process_neural_optimized(query, context)
                self._update_cache(cache_key, (result, "neural"))
                return result, "neural"

        except Exception as e:
            logger.error(f"Error during hybrid processing: {str(e)}")
            # Attempt symbolic processing as fallback
            result = self._process_symbolic(query)
            if result:
                return result, "symbolic"
            # Final fallback to neural processing
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
