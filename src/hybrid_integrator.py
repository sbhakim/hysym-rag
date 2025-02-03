# src/hybrid_integrator.py

import time
import logging
import torch
from typing import Tuple, List, Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HybridIntegrator:
    """
    A sophisticated integration system that combines symbolic and neural reasoning approaches
    while optimizing for performance and resource usage. The system includes intelligent
    caching, batch processing capabilities, and adaptive resource management.
    """

    def __init__(
            self,
            symbolic_reasoner,
            neural_retriever,
            resource_manager,
            query_expander=None,
            cache_ttl: int = 3600,  # Cache time-to-live in seconds
            batch_size: int = 4
    ):
        """
        Initialize the hybrid integration system with its core components.

        Args:
            symbolic_reasoner: Component for symbolic reasoning
            neural_retriever: Component for neural retrieval
            resource_manager: Manages system resources and performance
            query_expander: Optional component for query expansion
            cache_ttl: Time in seconds before cache entries expire
            batch_size: Size of batches for processing similar queries
        """
        self.symbolic_reasoner = symbolic_reasoner
        self.neural_retriever = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander

        # Enhanced cache with timestamps and metadata
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size

        # Performance tracking
        self.performance_stats = {
            'symbolic_hits': 0,
            'neural_hits': 0,
            'cache_hits': 0,
            'total_queries': 0
        }

        logger.info("HybridIntegrator initialized with enhanced caching and batch processing")

    def _generate_cache_key(self, query: str) -> str:
        """Generate a unique cache key for a query, accounting for expansions."""
        if self.query_expander:
            # Include expanded form in cache key to ensure consistency
            expanded = self.query_expander.expand_query(query)
            return f"{query}::{expanded}"
        return query

    def _get_valid_cache(self, cache_key: str) -> Optional[Tuple[List[str], str]]:
        """
        Retrieve a valid cache entry, checking for expiration.

        Args:
            cache_key: The key to look up in the cache

        Returns:
            Cached result if valid, None if expired or not found
        """
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self.cache_ttl):
                self.performance_stats['cache_hits'] += 1
                logger.debug(f"Cache hit for query: {cache_key}")
                return entry['result'], entry['source']
            else:
                # Clean up expired entry
                del self.cache[cache_key]
                logger.debug(f"Removed expired cache entry: {cache_key}")
        return None

    def _update_cache(self, cache_key: str, result: Tuple[List[str], str]) -> None:
        """
        Update the cache with a new result, managing memory usage.

        Args:
            cache_key: Key for caching the result
            result: Tuple of (result_list, source) to cache
        """
        # Implement simple LRU-style cache cleanup if needed
        if len(self.cache) > 1000:  # Arbitrary limit
            oldest_key = min(self.cache.items(), key=lambda x: x[1]['timestamp'])[0]
            del self.cache[oldest_key]

        self.cache[cache_key] = {
            'result': result[0],
            'source': result[1],
            'timestamp': datetime.now()
        }

    def process_query(self, query: str, context: str) -> Tuple[List[str], str]:
        """
        Process a query using the optimal combination of symbolic and neural approaches.

        This method implements a sophisticated decision process that:
        1. Checks for cached results
        2. Determines query complexity
        3. Chooses between symbolic and neural processing
        4. Handles fallbacks and error cases
        5. Manages system resources

        Args:
            query: The user's query string
            context: Additional context for neural processing

        Returns:
            Tuple of (result_list, source) where source is either "symbolic" or "neural"
        """
        self.performance_stats['total_queries'] += 1

        # Check cache first
        cache_key = self._generate_cache_key(query)
        cached_result = self._get_valid_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            # Determine query complexity
            query_complexity = (
                self.query_expander.get_query_complexity(query)
                if self.query_expander
                else 0.5
            )
            logger.info(f"Query complexity score: {query_complexity:.4f}")

            # Make resource-based decision
            use_symbolic = self.resource_manager.should_use_symbolic(query_complexity)

            if use_symbolic:
                logger.info("Using symbolic reasoning pathway")
                result = self._process_symbolic(query)
                if result:
                    self.performance_stats['symbolic_hits'] += 1
                    self._update_cache(cache_key, (result, "symbolic"))
                    return result, "symbolic"
                logger.info("Symbolic reasoning produced no valid result")

            # Neural processing with resource optimization
            logger.info("Falling back to neural reasoning pathway")
            result = self._process_neural_optimized(query, context)
            self.performance_stats['neural_hits'] += 1
            self._update_cache(cache_key, (result, "neural"))
            return result, "neural"

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Provide graceful fallback response
            fallback_response = ["An error occurred while processing your query. Please try again."]
            return fallback_response, "error"

    def _process_symbolic(self, query: str) -> Optional[List[str]]:
        """
        Process a query using symbolic reasoning with error handling.
        """
        try:
            symbolic_results = self.symbolic_reasoner.process_query(query)
            if symbolic_results and "No symbolic match found." not in symbolic_results:
                return symbolic_results
            return None
        except Exception as e:
            logger.error(f"Error in symbolic processing: {str(e)}")
            return None

    def _process_neural_optimized(self, query: str, context: str) -> List[str]:
        """
        Process a query using neural retrieval with optimized memory management.
        """
        # Clear GPU memory before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            start_time = time.time()
            result = [self.neural_retriever.retrieve_answer(context, query)]

            # Record performance metrics
            processing_time = time.time() - start_time
            self.resource_manager.neural_perf_times.append(processing_time)

            return result

        finally:
            # Ensure GPU memory is cleared after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Return performance metrics for monitoring and optimization.
        """
        total = max(1, self.performance_stats['total_queries'])
        return {
            'symbolic_ratio': self.performance_stats['symbolic_hits'] / total,
            'neural_ratio': self.performance_stats['neural_hits'] / total,
            'cache_hit_ratio': self.performance_stats['cache_hits'] / total,
            'average_neural_time': (
                sum(self.resource_manager.neural_perf_times) /
                len(self.resource_manager.neural_perf_times)
                if self.resource_manager.neural_perf_times
                else 0
            )
        }