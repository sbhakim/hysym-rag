# src/system/system_control_manager.py

import logging
import time
import heapq
import torch
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from src.resources.resource_manager import EnergyAwareScheduler

class SystemControlManager:
    """
    Manages overall system control flow, including error handling, resource management, and query processing.
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

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SystemControlManager")

        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'total_retries': 0
        }
        # Initialize energy-aware scheduler (with priority queue functionality)
        self.scheduler = EnergyAwareScheduler(self.resource_manager)

    def process_query_with_fallback(
            self,
            query: str,
            context: str,
            max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not isinstance(context, str):
            context = str(context)
        start_time = time.time()
        retries = 0
        max_retries = max_retries or self.error_retry_limit

        while retries <= max_retries:
            try:
                resource_status = self.resource_manager.check_resources()
                if not self._validate_resources(resource_status):
                    self.logger.warning("Resource limits exceeded, optimizing allocation...")
                    self.resource_manager.optimize_resources()

                # First, try to get a high-priority task from the scheduler's queue.
                scheduled_task = self.scheduler.schedule_next_task()
                if scheduled_task is None:
                    # Otherwise, use the default scheduling decision.
                    scheduled_task = self.scheduler.schedule_task("neural")

                if scheduled_task == "symbolic":
                    self.logger.info("Scheduled task: symbolic reasoning due to energy constraints.")
                    result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
                    reasoning_type = "symbolic"
                else:
                    self.logger.info("Scheduled task: neural reasoning.")
                    result = self._timed_process_query(query, context)
                    reasoning_type = "neural"

                self._update_metrics(start_time, True)
                return self.aggregator.format_response({
                    'result': result,
                    'processing_time': time.time() - start_time,
                    'resource_usage': resource_status,
                    'retries': retries,
                    'reasoning_type': reasoning_type
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
        try:
            if hasattr(self.hybrid_integrator, 'alignment_layer'):
                sym_dim = self.hybrid_integrator.alignment_layer.sym_adapter.in_features
                neural_dim = self.hybrid_integrator.alignment_layer.neural_adapter.in_features
                self.logger.debug(f"Alignment layer dimensions - Symbolic: {sym_dim}, Neural: {neural_dim}")
            start = time.time()

            def _check_timeout():
                if time.time() - start > self.max_query_time:
                    raise TimeoutError("Query processing exceeded maximum allowed time")

            self.logger.info("Attempting symbolic reasoning...")
            symbolic_result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
            _check_timeout()
            if symbolic_result:
                self.logger.info("Symbolic reasoning successful")
                return symbolic_result, "symbolic"

            self.logger.info("Symbolic reasoning insufficient, falling back to neural...")
            try:
                neural_result = self.hybrid_integrator.neural.retrieve_answer(context, query)
                _check_timeout()
                self.logger.info("Neural processing successful")
                return neural_result, "neural"
            except Exception as neural_error:
                self.logger.error(f"Neural processing failed: {str(neural_error)}")
                if symbolic_result:
                    return symbolic_result, "symbolic_fallback"
                raise

        except Exception as e:
            self.logger.error(f"Unexpected error in timed process: {str(e)}")
            raise

    def _validate_resources(self, status: Dict[str, float]) -> bool:
        cpu_threshold = 0.9  # 90%
        memory_threshold = 0.85  # 85%
        gpu_threshold = 0.95  # 95%
        return (
            status['cpu'] < cpu_threshold and
            status['memory'] < memory_threshold and
            status.get('gpu', 0) < gpu_threshold
        )

    def _handle_timeout(self):
        self.logger.info("Adjusting resource allocation due to timeout...")
        self.resource_manager.optimize_resources()
        time.sleep(1)

    def _handle_error(self, error: Exception):
        self.logger.info(f"Attempting recovery from error: {str(error)}")
        self.resource_manager.optimize_resources()
        time.sleep(0.5)

    def _handle_final_failure(
            self,
            start_time: float,
            error_msg: str
    ) -> Dict[str, Any]:
        self._update_metrics(start_time, False)
        return self.aggregator.format_response({
            'error': error_msg,
            'processing_time': time.time() - start_time,
            'resource_usage': self.resource_manager.check_resources(),
            'status': 'failed'
        })

    def _update_metrics(self, start_time: float, success: bool):
        try:
            response_time = time.time() - start_time
            self.performance_metrics['total_queries'] += 1
            if success:
                self.performance_metrics['successful_queries'] += 1
            else:
                self.performance_metrics['failed_queries'] += 1
            total_queries = self.performance_metrics['total_queries']
            current_avg = self.performance_metrics['avg_response_time']
            if total_queries == 1:
                new_avg = response_time
            else:
                new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
            self.performance_metrics['avg_response_time'] = new_avg
            self.logger.debug(
                f"Updated metrics - Success: {success}, Response Time: {response_time:.2f}s, New Avg: {new_avg:.2f}s"
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
        formatted = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success' if 'result' in response_data else 'error',
            'processing_time': response_data.get('processing_time', 0),
            'resource_usage': response_data.get('resource_usage', {}),
        }
        if 'result' in response_data:
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
