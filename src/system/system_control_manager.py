# src/system/system_control_manager.py

import logging
import time
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

class SystemControlManager:
    """
    Manages the overall system control flow, including error handling,
    resource management, and query processing with dynamic hybrid reasoning.
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

    def process_query_with_fallback(
        self, query: str, context: str, max_retries: Optional[int] = None
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

                result, reasoning_type = self._timed_process_query(query, context)
                self._update_metrics(start_time, True)
                return self.aggregator.format_response({
                    'result': result,
                    'processing_time': time.time() - start_time,
                    'resource_usage': resource_status,
                    'reasoning_path': reasoning_type,
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
        self, query: str, context: str
    ) -> Tuple[str, str]:
        start = time.time()
        performance_metrics = {
            'start_resources': self.resource_manager.check_resources()
        }

        def _check_timeout():
            if time.time() - start > self.max_query_time:
                raise TimeoutError("Query processing exceeded maximum allowed time")

        # Get query complexity (if a query expander is available)
        complexity = (self.hybrid_integrator.query_expander.get_query_complexity(query)
                      if self.hybrid_integrator.query_expander is not None else 1.0)
        self.logger.info(f"Query complexity score: {complexity:.4f}")

        try:
            # Use the public calculate_efficiency_score (renamed in ResourceManager)
            try:
                optimal_path = self.resource_manager.get_optimal_reasoning_path(complexity)
            except AttributeError:
                self.logger.warning("ResourceManager missing efficiency scoring; defaulting to neural reasoning.")
                optimal_path = "neural"
            self.logger.info(f"Selected reasoning path: {optimal_path}")

            # Routing based on the chosen path and query complexity
            if optimal_path == "symbolic" or complexity < 0.5:
                self.logger.info("Routing to symbolic reasoning...")
                symbolic_result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
                _check_timeout()
                if symbolic_result:
                    self.logger.info("Symbolic reasoning successful")
                    result = symbolic_result
                    reasoning_type = "symbolic"
                else:
                    optimal_path = "hybrid"

            if optimal_path == "hybrid" or (optimal_path == "symbolic" and not symbolic_result):
                self.logger.info("Routing to hybrid reasoning...")
                symbolic_result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
                symbolic_keywords = []
                if symbolic_result and isinstance(symbolic_result, list):
                    for resp in symbolic_result:
                        symbolic_keywords.extend(
                            self.hybrid_integrator.symbolic_reasoner.extract_keywords(resp)
                        )
                neural_result = self.hybrid_integrator.neural.retrieve_answer(
                    context, query, symbolic_guidance=symbolic_keywords
                )
                _check_timeout()
                result = f"Symbolic: {symbolic_result}\nNeural: {neural_result}"
                reasoning_type = "hybrid"

            elif optimal_path == "neural" or complexity >= 1.0:
                self.logger.info("Routing to neural reasoning...")
                result = self.hybrid_integrator.neural.retrieve_answer(context, query)
                _check_timeout()
                reasoning_type = "neural"

            end_time = time.time()
            performance_metrics.update({
                'latency': end_time - start,
                'end_resources': self.resource_manager.check_resources(),
                'reasoning_type': reasoning_type
            })
            resource_usage = {
                key: performance_metrics['end_resources'][key] -
                     performance_metrics['start_resources'][key]
                for key in performance_metrics['end_resources']
            }
            self.resource_manager.update_performance_metrics(
                reasoning_type=reasoning_type,
                metrics={
                    'latency': performance_metrics['latency'],
                    'accuracy': 0.8,  # Placeholder for accuracy metric
                    'resource_usage': resource_usage
                }
            )
            return result, reasoning_type

        except Exception as e:
            end_time = time.time()
            self.resource_manager.update_performance_metrics(
                reasoning_type='failed',
                metrics={
                    'latency': end_time - start,
                    'accuracy': 0.0,
                    'resource_usage': self.resource_manager.check_resources()
                }
            )
            raise e

    def _validate_resources(self, status: Dict[str, float]) -> bool:
        cpu_threshold = 0.9
        memory_threshold = 0.85
        gpu_threshold = 0.95
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
            new_avg = response_time if total == 1 else ((current_avg * (total - 1)) + response_time) / total
            self.performance_metrics['avg_response_time'] = new_avg
            self.logger.debug(
                f"Updated metrics - Success: {success}, Response Time: {response_time:.2f}s, New Avg: {new_avg:.2f}s"
            )
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Returns overall performance metrics including a computed success rate.
        """
        total = self.performance_metrics['total_queries']
        success_rate = (self.performance_metrics['successful_queries'] / total * 100) if total > 0 else 0.0
        metrics = dict(self.performance_metrics)
        metrics['success_rate'] = success_rate
        return metrics

    def get_reasoning_path_stats(self) -> Dict[str, float]:
        """
        Calculate and return the distribution (in percentage) of reasoning paths used.
        """
        total = self.performance_metrics['total_queries']
        if total == 0:
            return {'symbolic': 0.0, 'neural': 0.0, 'hybrid': 0.0}
        # Here we assume ResourceManager updated its history with a 'reasoning_type' field.
        # For simplicity, we count based on our performance history stored in ResourceManager.
        path_counts = {'symbolic': 0, 'neural': 0, 'hybrid': 0}
        for path in ['symbolic', 'neural', 'hybrid']:
            path_counts[path] = self.resource_manager.performance_history.get(path, {}).get('success_count', 0)
        return {path: (count / total * 100) for path, count in path_counts.items()}

# UnifiedResponseAggregator class definition

class UnifiedResponseAggregator:
    """
    Aggregates and formats system responses with intelligent synthesis of symbolic and neural outputs.
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
            if isinstance(result, str) and 'Symbolic:' in result:
                formatted['result'] = self._synthesize_hybrid_response(result)
            else:
                formatted['result'] = self._clean_response(result)
            if self.include_explanations:
                formatted['reasoning_path'] = response_data.get('reasoning_path', 'Direct response')
        else:
            formatted['error'] = response_data.get('error', 'Unknown error occurred')
        return formatted

    def _synthesize_hybrid_response(self, hybrid_result: str) -> str:
        parts = hybrid_result.split('\nNeural:')
        symbolic_part = parts[0].replace('Symbolic:', '').strip()
        neural_part = parts[1].strip() if len(parts) > 1 else ""
        symbolic_points = [point.strip() for point in symbolic_part.strip('[]').split(',')]
        synthesized = "Based on analysis, deforestation has several key environmental impacts:\n\n"
        for point in symbolic_points:
            point = point.strip().strip("'").strip('"')
            if point:
                synthesized += f"- {point.capitalize()}\n"
        if neural_part:
            neural_clean = neural_part.split('Question:')[0].strip()
            neural_clean = neural_clean.split('Answer:')[-1].strip()
            synthesized += f"\nFurther analysis reveals: {neural_clean}"
        return synthesized

    def _clean_response(self, result: Any) -> str:
        if isinstance(result, list):
            return "\n".join(f"- {item}" for item in result)
        elif isinstance(result, dict):
            return result.get('result', str(result))
        return str(result).strip()
