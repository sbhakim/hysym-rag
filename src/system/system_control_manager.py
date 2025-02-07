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
            self,
            query: str,
            context: str
    ) -> Tuple[str, str]:
        start = time.time()
        def _check_timeout():
            if time.time() - start > self.max_query_time:
                raise TimeoutError("Query processing exceeded maximum allowed time")

        # Use the query_expander from the hybrid integrator if available.
        if self.hybrid_integrator.query_expander is not None:
            complexity = self.hybrid_integrator.query_expander.get_query_complexity(query)
        else:
            complexity = 1.0  # default high complexity
        self.logger.info(f"Query complexity score: {complexity:.4f}")

        if complexity < 0.5:
            self.logger.info("Routing to symbolic reasoning...")
            symbolic_result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
            _check_timeout()
            if symbolic_result:
                self.logger.info("Symbolic reasoning successful")
                return symbolic_result, "symbolic"
        elif complexity < 1.0:
            self.logger.info("Routing to hybrid reasoning...")
            symbolic_result = self.hybrid_integrator.symbolic_reasoner.process_query(query)
            symbolic_keywords = []
            if symbolic_result and isinstance(symbolic_result, list):
                for resp in symbolic_result:
                    # Here we call extract_keywords on the symbolic reasoner;
                    # if that class (e.g., GraphSymbolicReasoner) lacks it, it must be added.
                    symbolic_keywords.extend(self.hybrid_integrator.symbolic_reasoner.extract_keywords(resp))
            neural_result = self.hybrid_integrator.neural.retrieve_answer(context, query, symbolic_guidance=symbolic_keywords)
            _check_timeout()
            combined = f"Symbolic: {symbolic_result}\nNeural: {neural_result}"
            return combined, "hybrid"
        else:
            self.logger.info("Routing to neural reasoning...")
            neural_result = self.hybrid_integrator.neural.retrieve_answer(context, query)
            _check_timeout()
            return neural_result, "neural"

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
            total = self.performance_metrics['total_queries']
            current_avg = self.performance_metrics['avg_response_time']
            new_avg = response_time if total == 1 else ((current_avg * (total - 1)) + response_time) / total
            self.performance_metrics['avg_response_time'] = new_avg
            self.logger.debug(f"Updated metrics - Success: {success}, Response Time: {response_time:.2f}s, New Avg: {new_avg:.2f}s")
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")


class UnifiedResponseAggregator:
    """
    Aggregates and formats system responses with intelligent synthesis of
    symbolic and neural outputs.
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
            # Extract the symbolic and neural parts if this is a hybrid result
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
        # Split into symbolic and neural parts
        parts = hybrid_result.split('\nNeural:')
        symbolic_part = parts[0].replace('Symbolic:', '').strip()
        neural_part = parts[1].strip() if len(parts) > 1 else ""

        # Extract symbolic insights
        symbolic_points = [point.strip() for point in symbolic_part.strip('[]').split(',')]

        # Build a coherent response
        synthesized = "Based on analysis, deforestation has several key environmental impacts:\n\n"

        # Add symbolic insights first
        for point in symbolic_points:
            point = point.strip().strip("'").strip('"')
            if point:
                synthesized += f"- {point.capitalize()}\n"

        # Add neural elaboration if available
        if neural_part:
            # Clean up neural response by removing question patterns
            neural_clean = neural_part.split('Question:')[0].strip()
            neural_clean = neural_clean.split('Answer:')[-1].strip()

            synthesized += f"\nFurther analysis reveals: {neural_clean}"

        return synthesized

    def _clean_response(self, result: Any) -> str:
        """
        Clean and format a single-source response.
        """
        if isinstance(result, list):
            return "\n".join(f"- {item}" for item in result)
        elif isinstance(result, dict):
            return result.get('result', str(result))
        return str(result).strip()
