# src/system/system_control_manager.py

import logging
import time
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

class SystemControlManager:
    """Manages system control flow, including error handling, resource management, and query processing."""

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
        # New: Track the distribution of reasoning paths
        self.reasoning_path_stats = {
            "symbolic": 0,
            "hybrid": 0,
            "neural": 0,
            "fallback": 0
        }

    def process_query_with_fallback(
        self, query: str, context: str, max_retries: Optional[int] = None, forced_path: Optional[str] = None
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

                result, reasoning_type = self._timed_process_query(query, context, forced_path)
                self._update_metrics(start_time, True)
                return self.aggregator.format_response({
                    'result': result,
                    'processing_time': time.time() - start_time,
                    'resource_usage': resource_status,
                    'reasoning_path': reasoning_type,
                    'retries': retries
                })
            except TimeoutError:
                self.logger.error("Query processing timeout.")
                retries += 1
                if retries <= max_retries:
                    self._handle_timeout()
                    continue
                return self._handle_final_failure(start_time, "Timeout")
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                retries += 1
                if retries <= max_retries:
                    self._handle_error(e)
                    continue
                return self._handle_final_failure(start_time, str(e))

    def _timed_process_query(
        self, query: str, context: str, forced_path: Optional[str] = None
    ) -> Tuple[str, str]:
        start = time.time()

        def _check_timeout():
            if time.time() - start > self.max_query_time:
                raise TimeoutError("Query processing exceeded maximum time")

        complexity = self.hybrid_integrator.query_expander.get_query_complexity(query) if self.hybrid_integrator.query_expander else 1.0
        self.logger.info(f"Query complexity score: {complexity:.4f}")

        try:
            optimal_path = forced_path or self.resource_manager.get_optimal_reasoning_path(complexity)
            self.logger.info(f"Selected reasoning path: {optimal_path}")

            if optimal_path == "symbolic" or complexity < 0.5:
                self.logger.info("Routing to symbolic reasoning...")
                self.reasoning_path_stats["symbolic"] += 1
                symbolic_result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
                _check_timeout()
                if symbolic_result:
                    return symbolic_result, "symbolic"
                optimal_path = "hybrid"

            if optimal_path == "hybrid":
                self.logger.info("Routing to hybrid reasoning...")
                self.reasoning_path_stats["hybrid"] += 1
                symbolic_result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
                symbolic_keywords = []
                if symbolic_result and isinstance(symbolic_result, list):
                    symbolic_keywords = [kw for resp in symbolic_result for kw in self.hybrid_integrator.symbolic_reasoner.extract_keywords(resp)]
                neural_result = self.hybrid_integrator.neural.retrieve_answer(context, query, symbolic_guidance=symbolic_keywords)
                _check_timeout()
                return f"Symbolic: {symbolic_result}\nNeural: {neural_result}", "hybrid"

            self.logger.info("Routing to neural reasoning...")
            self.reasoning_path_stats["neural"] += 1
            result = self.hybrid_integrator.neural.retrieve_answer(context, query)
            _check_timeout()
            return result, "neural"

        except Exception as e:
            self.resource_manager.update_performance_metrics('failed', {'latency': time.time() - start, 'accuracy': 0.0})
            self.reasoning_path_stats["fallback"] += 1
            raise e

    def _validate_resources(self, status: Dict[str, float]) -> bool:
        return status['cpu'] < 0.9 and status['memory'] < 0.85 and status.get('gpu', 0) < 0.95

    def _handle_timeout(self):
        self.logger.info("Adjusting resource allocation due to timeout...")
        self.resource_manager.optimize_resources()
        time.sleep(1)

    def _handle_error(self, error: Exception):
        self.logger.info(f"Attempting recovery from error: {error}")
        self.resource_manager.optimize_resources()
        time.sleep(0.5)

    def _handle_final_failure(self, start_time: float, error_msg: str) -> Dict[str, Any]:
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
            total = self.performance_metrics['total_queries']
            current_avg = self.performance_metrics['avg_response_time']
            new_avg = (current_avg * (total - 1) + response_time) / total if total > 1 else response_time
            self.performance_metrics['avg_response_time'] = new_avg
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def get_performance_metrics(self) -> Dict[str, float]:
        total = self.performance_metrics['total_queries']
        success_rate = (self.performance_metrics['successful_queries'] / total * 100) if total > 0 else 0.0
        return {**self.performance_metrics, 'success_rate': success_rate}

    def get_reasoning_path_stats(self) -> Dict[str, float]:
        total = self.performance_metrics['total_queries']
        if total == 0:
            return self.reasoning_path_stats
        # Return percentage distribution for each reasoning path
        return {path: (count / total * 100) for path, count in self.reasoning_path_stats.items()}

class UnifiedResponseAggregator:
    """Aggregates and formats system responses with intelligent synthesis of symbolic and neural outputs."""

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
            formatted['result'] = self._clean_response(response_data['result'])
            if self.include_explanations:
                formatted['reasoning_path'] = response_data.get('reasoning_path', 'Direct response')
        else:
            formatted['error'] = response_data.get('error', 'Unknown error')
        return formatted

    def _clean_response(self, result: Any) -> str:
        if isinstance(result, list):
            return "\n".join(f"- {item}" for item in result)
        if isinstance(result, dict):
            return result.get('result', str(result))
        return str(result).strip()
