# src/system/system_control_manager.py

import logging
import time
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

class SystemControlManager:
    """
    Manages the overall system control flow, including error handling,
    resource management, and query processing.
    """

    def __init__(
            self,
            hybrid_integrator,
            resource_manager,
            aggregator,
            error_retry_limit: int = 2,
            max_query_time: float = 10.0
    ):
        self.hybrid_integrator = hybrid_integrator
        self.resource_manager = resource_manager
        self.aggregator = aggregator
        self.error_retry_limit = error_retry_limit
        self.max_query_time = max_query_time

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SystemControlManager")

        # Track performance metrics
        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'total_retries': 0
        }

    def process_query_with_fallback(
            self,
            query: str,
            context: str,
            max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a query with automatic fallback mechanisms and error handling.

        Args:
            query: The input query string.
            context: Additional context for processing.
            max_retries: Optional override for retry limit.

        Returns:
            Dict containing the response and metadata.
        """
        # Validate inputs
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        if not isinstance(context, str):
            context = str(context)  # Convert to string if possible
        start_time = time.time()
        retries = 0
        max_retries = max_retries or self.error_retry_limit

        while retries <= max_retries:
            try:
                # Monitor resources before processing
                resource_status = self.resource_manager.check_resources()
                if not self._validate_resources(resource_status):
                    self.logger.warning("Resource limits exceeded, optimizing allocation...")
                    self.resource_manager.optimize_resources()

                # Process query with timing constraints
                result = self._timed_process_query(query, context)

                # Update performance metrics
                self._update_metrics(start_time, True)

                return self.aggregator.format_response({
                    'result': result,
                    'processing_time': time.time() - start_time,
                    'resource_usage': resource_status,
                    'retries': retries
                })

            except TimeoutError as te:
                self.logger.error(f"Query processing timeout: {str(te)}")
                retries += 1
                if retries <= max_retries:
                    self._handle_timeout()
                    continue
                return self._handle_final_failure(start_time, "Timeout")

            except Exception as e:
                self.logger.error(f"Error processing query: {str(e)}")
                retries += 1
                if retries <= max_retries:
                    self._handle_error(e)
                    continue
                return self._handle_final_failure(start_time, str(e))

    def _timed_process_query(
            self,
            query: str,
            context: str
    ) -> Tuple[str, str]:
        """
        Process a query with a timeout constraint and dimension validation.
        Returns a tuple of (result, reasoning_type).
        """
        try:
            # Validate hybrid integrator's alignment layer dimensions
            if hasattr(self.hybrid_integrator, 'alignment_layer'):
                sym_dim = self.hybrid_integrator.alignment_layer.sym_adapter.in_features
                neural_dim = self.hybrid_integrator.alignment_layer.neural_adapter.in_features
                self.logger.debug(f"Alignment layer dimensions - Symbolic: {sym_dim}, Neural: {neural_dim}")
            start = time.time()

            def _check_timeout():
                if time.time() - start > self.max_query_time:
                    raise TimeoutError("Query processing exceeded maximum allowed time")

            try:
                # First attempt with symbolic reasoning.
                # Note: Changed to use process_query() since GraphSymbolicReasoner does not define process()
                self.logger.info("Attempting symbolic reasoning...")
                symbolic_result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
                _check_timeout()

                if symbolic_result:
                    self.logger.info("Symbolic reasoning successful")
                    return symbolic_result, "symbolic"

                # Fallback to neural if symbolic fails.
                self.logger.info("Symbolic reasoning insufficient, falling back to neural...")
                try:
                    neural_result = self.hybrid_integrator.neural.retrieve_answer(context, query)
                    _check_timeout()
                    self.logger.info("Neural processing successful")
                    return neural_result, "neural"
                except Exception as neural_error:
                    self.logger.error(f"Neural processing failed: {str(neural_error)}")
                    # If we have any symbolic result, return it as fallback
                    if symbolic_result:
                        return symbolic_result, "symbolic_fallback"
                    raise

            except Exception as e:
                self.logger.error(f"Error in timed process: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Unexpected error in timed process: {str(e)}")
            raise

    def _validate_resources(self, status: Dict[str, float]) -> bool:
        """
        Validate resource usage against thresholds.
        """
        cpu_threshold = 0.9  # 90% CPU usage limit.
        memory_threshold = 0.85  # 85% memory usage limit.
        gpu_threshold = 0.95  # 95% GPU usage limit.

        return (
            status['cpu'] < cpu_threshold and
            status['memory'] < memory_threshold and
            status.get('gpu', 0) < gpu_threshold
        )

    def _handle_timeout(self):
        """
        Handle timeout scenarios by adjusting resource allocation.
        """
        self.logger.info("Adjusting resource allocation due to timeout...")
        self.resource_manager.optimize_resources()
        time.sleep(1)  # Brief pause before retry.

    def _handle_error(self, error: Exception):
        """
        Handle general processing errors.
        """
        self.logger.info(f"Attempting recovery from error: {str(error)}")
        self.resource_manager.optimize_resources()
        time.sleep(0.5)

    def _handle_final_failure(
            self,
            start_time: float,
            error_msg: str
    ) -> Dict[str, Any]:
        """
        Handle the case when all retries are exhausted.
        """
        self._update_metrics(start_time, False)
        return self.aggregator.format_response({
            'error': error_msg,
            'processing_time': time.time() - start_time,
            'resource_usage': self.resource_manager.check_resources(),
            'status': 'failed'
        })

    def _update_metrics(self, start_time: float, success: bool):
        """
        Update internal performance metrics with proper error handling.
        """
        try:
            # Calculate response time first.
            response_time = time.time() - start_time

            # Update query counts.
            self.performance_metrics['total_queries'] += 1
            if success:
                self.performance_metrics['successful_queries'] += 1
            else:
                self.performance_metrics['failed_queries'] += 1

            # Get current metrics.
            total_queries = self.performance_metrics['total_queries']
            current_avg = self.performance_metrics['avg_response_time']

            # Calculate new average - handle first query case.
            if total_queries == 1:
                new_avg = response_time
            else:
                new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries

            # Update the metrics.
            self.performance_metrics['avg_response_time'] = new_avg

            self.logger.debug(
                f"Updated metrics - Success: {success}, "
                f"Response Time: {response_time:.2f}s, "
                f"New Avg: {new_avg:.2f}s"
            )
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

class UnifiedResponseAggregator:
    """
    Aggregates and formats system responses consistently.
    """

    def __init__(self, include_explanations: bool = False):
        self.include_explanations = include_explanations

    def format_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the response with a consistent structure and optional explanations.
        """
        formatted = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success' if 'result' in response_data else 'error',
            'processing_time': response_data.get('processing_time', 0),
            'resource_usage': response_data.get('resource_usage', {}),
        }

        if 'result' in response_data:
            # Ensure the result is a string.
            result = response_data['result']
            if isinstance(result, dict) and 'result' in result:
                result = result['result']
            elif not isinstance(result, str):
                result = str(result)
            formatted['result'] = result
            if self.include_explanations:
                formatted['reasoning_path'] = response_data.get('reasoning_path', 'Direct response')
        else:
            formatted['error'] = response_data.get('error', 'Unknown error occurred')

        return formatted
