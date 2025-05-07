# src/system/system_control_manager.py

import logging
import time
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
import torch  # Not directly used in this file but often present in PyTorch projects
import numpy as np
from collections import defaultdict
from src.utils.metrics_collector import MetricsCollector


# Assuming HybridIntegrator might be type hinted or for isinstance checks later if needed
# from src.integrators.hybrid_integrator import HybridIntegrator


class UnifiedResponseAggregator:
    """
    Aggregator for academic evaluation responses with detailed reasoning.
    """

    def __init__(self, include_explanations: bool = True):
        self.include_explanations = include_explanations
        self.logger = logging.getLogger("UnifiedResponseAggregator")  # Added logger initialization

    def format_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure 'result' is present, especially for DROP which might return structured dicts
        if 'result' not in data:
            self.logger.warning("Formatting response: 'result' key missing from data.")
            data['result'] = data.get('error', "No result available")  # Default to error or a placeholder

        if self.include_explanations:
            data.setdefault("explanation", self._generate_detailed_explanation(data))
        return data


    def _generate_detailed_explanation(self, data: Dict[str, Any]) -> str:
        """Generate detailed reasoning explanation for academic analysis."""
        parts = []
        if 'reasoning_path' in data:  # This is the string like 'symbolic', 'hybrid'
            parts.append(f"Reasoning Approach: {data['reasoning_path']}")

        # For DROP, data['result'] might be a dict. For HotpotQA, it might be a string.
        # We probably don't want to print the whole structured result in the explanation.
        # The main result should be handled by the caller.
        # This explanation focuses on metadata.

        if 'processing_time' in data:
            parts.append(f"Processing Time: {data['processing_time']:.3f}s")
        if 'resource_usage' in data:
            parts.append("Resource Utilization:")
            for resource, usage in data['resource_usage'].items():
                # Ensure usage is a number before formatting
                if isinstance(usage, (int, float)):
                    parts.append(f"- {resource.capitalize()}: {usage * 100:.1f}%")
                else:
                    parts.append(f"- {resource.capitalize()}: {usage}")  # Print as is if not a number

        if data.get('retries', 0) > 0:
            parts.append(f"Retries Attempted: {data['retries']}")

        return " | ".join(parts) if parts else "No additional explanation metadata provided."


class SystemControlManager:
    """
    Enhanced SystemControlManager for academic evaluation of HySym-RAG.
    Focuses on reproducible metrics, detailed analysis, and academic logging.
    """

    def __init__(
            self,
            hybrid_integrator,  # Should be an instance of HybridIntegrator
            resource_manager,
            aggregator,  # Should be an instance of UnifiedResponseAggregator
            metrics_collector: Optional[MetricsCollector] = None,
            error_retry_limit: int = 2,
            max_query_time: float = 10.0,  # Default, consider increasing for complex tasks
            performance_window: int = 100  # For rolling metrics, not heavily used here yet
    ):
        self.hybrid_integrator = hybrid_integrator
        self.resource_manager = resource_manager
        self.aggregator = aggregator
        self.error_retry_limit = error_retry_limit
        self.max_query_time = max_query_time  # Currently not enforced with a timeout mechanism here
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger("SystemControlManager")
        self.logger.setLevel(logging.INFO)

        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'path_performance': defaultdict(list),  # Stores lists of response times per path
            # 'resource_efficiency' and 'reasoning_quality' are more complex and likely aggregated
            # by MetricsCollector from detailed per-query resource_usage and reasoning_path_details
        }
        self.path_history = []  # Stores dicts about path selection decisions

        # Adaptive thresholds for path determination
        self.adaptive_thresholds = {
            'complexity_threshold': 0.75,  # Queries below this might go symbolic
            'resource_pressure_threshold': 0.80,  # If system resource usage above this, might favor symbolic
            'efficiency_threshold': 0.6  # Not directly used in path determination logic here
        }

    def process_query_with_fallback(
            self,
            query: str,
            context: str,
            max_retries: Optional[int] = None,
            forced_path: Optional[str] = None,
            query_complexity: Optional[float] = None,
            query_id: Optional[str] = None  # <-- ADDED PARAMETER
    ) -> Dict[str, Any]:
        """
        Process query with enhanced error handling and metrics collection.
        Now accepts and forwards query_id.
        """
        if not isinstance(query, str) or not query.strip():
            # Consider raising ValueError for main.py to catch or return a specific error structure
            self.logger.error(f"Invalid query provided (query_id: {query_id}): Must be a non-empty string.")
            return self._handle_final_failure(time.time(), "Invalid query string", query_id, is_error_object=True)

        start_time = time.time()
        retries = 0
        max_retries_effective = max_retries if max_retries is not None else self.error_retry_limit

        self.logger.info(f"Processing query_id: {query_id}, Query: '{query[:100]}...'")

        # Record initial resource state for delta calculation
        initial_resources = self.resource_manager.check_resources()

        final_result_obj: Any = None
        reasoning_path_taken: str = "unknown"

        while retries <= max_retries_effective:
            try:
                resource_status = self.resource_manager.check_resources()
                if not self._validate_resources(resource_status):
                    self.logger.warning(f"Resource limits exceeded (CPU: {resource_status.get('cpu', 0):.2f}, "
                                        f"Mem: {resource_status.get('memory', 0):.2f}, GPU: {resource_status.get('gpu', 0):.2f}). "
                                        f"Optimizing allocation for query_id: {query_id}...")
                    self._optimize_resources()  # Placeholder for actual optimization logic

                # query_complexity is calculated once before the loop if not provided.
                # If it's None initially, _timed_process_query will calculate it.
                current_query_complexity = query_complexity
                if current_query_complexity is None and hasattr(self.hybrid_integrator,
                                                                "query_expander") and self.hybrid_integrator.query_expander:
                    current_query_complexity = self.hybrid_integrator.query_expander.get_query_complexity(query)
                    self.logger.info(
                        f"Calculated query complexity for query_id {query_id}: {current_query_complexity:.4f}")
                elif current_query_complexity is None:
                    current_query_complexity = 0.8  # Default if still None

                final_result_obj, reasoning_path_taken = self._timed_process_query(
                    query,
                    context,
                    forced_path,
                    current_query_complexity,  # Use the potentially calculated complexity
                    query_id  # Pass query_id
                )

                # Check if the result indicates an internal processing error from lower levels
                if isinstance(final_result_obj, dict) and final_result_obj.get("type") == "error_object":
                    raise RuntimeError(
                        f"Downstream processing error for query_id {query_id}: {final_result_obj.get('error')}")

                processing_time_seconds = time.time() - start_time
                final_utilized_resources = self.resource_manager.check_resources()
                resource_delta = self._calculate_resource_delta(initial_resources, final_utilized_resources)

                self._update_overall_metrics(start_time, True, reasoning_path_taken, query_id)  # Mark as success

                # Collect academic metrics
                if self.metrics_collector:
                    # For DROP, final_result_obj is structured. For text, it's a string.
                    # MetricsCollector.collect_query_metrics expects a string prediction for some of its generic logic.
                    prediction_for_metrics = str(final_result_obj)  # Default to string representation
                    if self.hybrid_integrator.dataset_type == 'drop' and isinstance(final_result_obj, dict):
                        # Create a concise string representation for general logging if needed,
                        # but primary evaluation uses the object.
                        if 'number' in final_result_obj:
                            prediction_for_metrics = f"DROP_Num: {final_result_obj['number']}"
                        elif 'spans' in final_result_obj:
                            prediction_for_metrics = f"DROP_Spans: {str(final_result_obj['spans'])[:50]}"
                        elif 'date' in final_result_obj:
                            prediction_for_metrics = f"DROP_Date: {final_result_obj['date']}"
                        elif 'error' in final_result_obj:
                            prediction_for_metrics = f"DROP_Error: {final_result_obj['error']}"

                    self.metrics_collector.collect_query_metrics(
                        query=query,  # Consider adding query_id here too if MC schema supports
                        prediction=prediction_for_metrics,
                        ground_truth=None,  # Ground truth is handled by Evaluation class later
                        reasoning_path=reasoning_path_taken,
                        processing_time=processing_time_seconds,
                        resource_usage=resource_delta,
                        complexity_score=current_query_complexity
                    )

                # Format response using aggregator
                # The 'result' key here will hold the final_result_obj (string or dict for DROP)
                return self.aggregator.format_response({
                    'query_id': query_id,
                    'result': final_result_obj,
                    'processing_time': processing_time_seconds,
                    'resource_usage': resource_delta,
                    'reasoning_path': reasoning_path_taken,
                    'retries': retries
                })

            except TimeoutError as e:  # This would require an actual timeout mechanism implementation
                self.logger.error(f"Query processing timeout for query_id {query_id}: {str(e)}")
                retries += 1
                if retries > max_retries_effective:
                    return self._handle_final_failure(start_time, "Timeout", query_id, is_error_object=True)
                self.logger.info(f"Retrying query_id {query_id} ({retries}/{max_retries_effective})...")
                time.sleep(0.5 * retries)  # Basic backoff
                continue

            except Exception as e:
                self.logger.exception(
                    f"Error processing query_id {query_id} (attempt {retries + 1}/{max_retries_effective + 1}): {str(e)}")
                retries += 1
                if retries > max_retries_effective:
                    return self._handle_final_failure(start_time, str(e), query_id, is_error_object=True)
                self.logger.info(f"Retrying query_id {query_id} ({retries}/{max_retries_effective})...")
                time.sleep(0.5 * retries)  # Basic backoff
                continue

        # Should not be reached if loop exits correctly, but as a safeguard:
        return self._handle_final_failure(start_time, "Max retries exceeded without explicit error type.", query_id,
                                          is_error_object=True)

    def _timed_process_query(
            self,
            query: str,
            context: str,
            forced_path: Optional[str] = None,
            query_complexity: Optional[float] = None,  # Should be pre-calculated now
            query_id: Optional[str] = None  # <-- ADDED PARAMETER
    ) -> Tuple[Any, str]:  # Return type is Any for result_obj
        """
        Process query with enhanced timing and path selection logging.
        Accepts query_id and passes it along.
        """
        path_selection_start_time = time.time()

        # Use pre-calculated query_complexity if available
        current_query_complexity = query_complexity
        if current_query_complexity is None:  # Fallback if not passed, though main.py should pass it
            if hasattr(self.hybrid_integrator, "query_expander") and self.hybrid_integrator.query_expander:
                current_query_complexity = self.hybrid_integrator.query_expander.get_query_complexity(query)
                self.logger.info(f"Re-calculated query complexity for qid {query_id}: {current_query_complexity:.4f}")
            else:
                current_query_complexity = 0.8  # Default fallback
                self.logger.warning(
                    f"Query complexity not provided or calculable for qid {query_id}, using default: {current_query_complexity}")

        if forced_path:
            optimal_path = forced_path
            self.logger.info(f"Forced reasoning path for qid {query_id}: {optimal_path}")
        else:
            optimal_path = self._determine_optimal_path(
                current_query_complexity,  # Use the resolved complexity
                self.resource_manager.check_resources(),
                query_id
            )
            # _determine_optimal_path already logs

        self.logger.debug(
            f"Path selection for qid {query_id} took: {time.time() - path_selection_start_time:.4f}s. Path: {optimal_path}")

        # Execute along the determined path
        execution_start_time = time.time()
        result_obj = self._execute_processing_path(
            optimal_path,
            query,
            context,
            current_query_complexity,  # Pass complexity to execution
            query_id  # Pass query_id
        )
        path_execution_time = time.time() - execution_start_time
        self.logger.debug(f"Path execution for qid {query_id} on '{optimal_path}' took: {path_execution_time:.4f}s")

        self._update_path_specific_metrics(optimal_path, path_execution_time, query_id)
        return result_obj, optimal_path

    def _determine_optimal_path(
            self,
            query_complexity: float,
            resource_status: Dict[str, float],
            query_id: Optional[str] = None  # For logging
    ) -> str:
        """
        Determine optimal processing path based on query complexity and resource status.
        """
        # Ensure GPU usage is 0 if no GPU is available or tracked
        gpu_usage = resource_status.get('gpu', 0.0) if self.resource_manager._gpu_available else 0.0

        resource_pressure = max(
            resource_status.get('cpu', 0.0),
            resource_status.get('memory', 0.0),
            gpu_usage
        )

        decision_factors = {
            'query_complexity': round(query_complexity, 4),
            'cpu_usage': round(resource_status.get('cpu', 0.0), 4),
            'memory_usage': round(resource_status.get('memory', 0.0), 4),
            'gpu_usage': round(gpu_usage, 4),
            'overall_resource_pressure': round(resource_pressure, 4)
        }

        path = "hybrid"  # Default path
        reason = "Default to hybrid; conditions for symbolic not met."

        # Logic for path selection:
        # Prefer symbolic if query complexity is low OR if resource pressure is high.
        # This needs to be adapted based on whether the symbolic reasoner can handle the dataset_type
        can_symbolic_handle_dataset = True  # Assume true by default
        if self.hybrid_integrator.dataset_type == 'drop':
            # Check if symbolic reasoner is actually equipped for DROP
            # This might involve a check on symbolic_reasoner's capabilities
            if not hasattr(self.hybrid_integrator.symbolic_reasoner, 'process_drop_query'):  # Example check
                self.logger.warning(
                    f"Symbolic reasoner may not be fully equipped for DROP. Path selection might default to hybrid/neural for qid {query_id}.")
                # can_symbolic_handle_dataset = False # If symbolic cannot do DROP, don't choose it based on complexity/pressure alone

        if can_symbolic_handle_dataset:
            if query_complexity < self.adaptive_thresholds['complexity_threshold']:
                path = "symbolic"
                reason = f"Low query complexity ({query_complexity:.2f} < {self.adaptive_thresholds['complexity_threshold']}) favors symbolic."
            elif resource_pressure > self.adaptive_thresholds['resource_pressure_threshold']:
                path = "symbolic"
                reason = f"High resource pressure ({resource_pressure:.2f} > {self.adaptive_thresholds['resource_pressure_threshold']}) favors symbolic."
            else:
                path = "hybrid"  # Explicitly state hybrid if conditions for symbolic aren't met
                reason = "Balanced complexity and resource usage favor hybrid approach."
        else:  # If symbolic cannot handle this dataset type
            path = "hybrid"  # Or "neural" if hybrid also can't handle it well without symbolic
            reason = f"Symbolic path not fully supported for {self.hybrid_integrator.dataset_type}; defaulting to hybrid/neural."

        self.logger.info(
            f"Path Selection for QID {query_id} - Factors: {decision_factors} -> Selected Path: {path} (Reason: {reason})")

        self.path_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_id': query_id,
            'path': path,
            'factors': decision_factors,
            'reason': reason
        })
        return path

    def _execute_processing_path(
            self,
            path: str,
            query: str,
            context: str,
            query_complexity: float,  # Now consistently available
            query_id: Optional[str] = None  # <-- ADDED PARAMETER
    ) -> Any:  # Return type is Any, as DROP might return structured objects
        """
        Execute query processing along the selected path.
        Now accepts and forwards query_id.
        """
        execution_path_start_time = time.time()
        result_obj: Any = None  # Default result object
        component_success = True  # Assume success initially

        try:
            if path == "symbolic":
                # Symbolic reasoner might need context and query_id for dataset-specific logic (e.g., DROP)
                result_obj = self.hybrid_integrator.symbolic_reasoner.process_query(
                    query,
                    context=context,  # Pass context
                    dataset_type=self.hybrid_integrator.dataset_type,  # Pass dataset type
                    query_id=query_id  # Pass query_id
                )
            elif path == "neural":
                # Neural retriever might also benefit from dataset_type for prompting
                result_obj = self.hybrid_integrator.neural_retriever.retrieve_answer(
                    context,
                    query,
                    query_complexity=query_complexity,
                    dataset_type=self.hybrid_integrator.dataset_type  # Pass dataset type
                    # query_id is not explicitly in retrieve_answer signature yet, add if needed for logging/cache there
                )
            elif path == "hybrid":  # Default/Hybrid path
                # HybridIntegrator's process_query is the main entry point for hybrid logic
                # It now also accepts query_id
                result_obj, _ = self.hybrid_integrator.process_query(  # Unpack, we only need the result object here
                    query,
                    context,
                    query_complexity=query_complexity,
                    query_id=query_id,  # Pass query_id
                    # supporting_facts are not directly available to SCM, HI gets them if needed via main.py
                )
            else:
                self.logger.error(f"Unknown processing path for qid {query_id}: {path}. Defaulting to error.")
                result_obj = "Error: Unknown processing path."
                if self.hybrid_integrator.dataset_type == 'drop':
                    result_obj = {"error": f"Unknown processing path: {path}", "type": "error_object"}
                component_success = False

            # Check for error objects from downstream
            if isinstance(result_obj, dict) and result_obj.get("type") == "error_object":
                self.logger.error(
                    f"Error object received from path '{path}' for qid {query_id}: {result_obj.get('error')}")
                component_success = False  # Mark as failure for metrics if an error object is returned

        except Exception as e:
            self.logger.exception(f"Exception during path execution '{path}' for qid {query_id}: {str(e)}")
            result_obj = f"Error during {path} execution."
            if self.hybrid_integrator.dataset_type == 'drop':
                result_obj = {"error": f"Exception in {path} path: {str(e)}", "type": "error_object"}
            component_success = False
            # Re-raise to be caught by the main retry loop in process_query_with_fallback
            raise e
        finally:
            path_component_time = time.time() - execution_path_start_time
            if self.metrics_collector:
                # For hybrid, component_times in HybridIntegrator are used.
                # For direct symbolic/neural, log here.
                if path in ["symbolic", "neural"]:
                    self.metrics_collector.collect_component_metrics(
                        component=path,
                        execution_time=path_component_time,
                        success=component_success,  # Based on whether an exception occurred or error object returned
                        error_rate=0.0 if component_success else 1.0,
                        resource_usage=self.resource_manager.check_resources()  # Snapshot after execution
                    )

            # Add timing to the result object if it's a dictionary (common for hybrid/DROP)
            # This is more for debugging path execution time itself. Overall time is handled by caller.
            if isinstance(result_obj, dict):
                result_obj['path_execution_time_ms'] = round(path_component_time * 1000, 2)

        if result_obj is None:  # Safeguard if no path was taken or result was not set
            self.logger.error(f"Execution path '{path}' for qid {query_id} resulted in None. Returning error object.")
            result_obj = "Error: No result from processing path."
            if self.hybrid_integrator.dataset_type == 'drop':
                result_obj = {"error": "No result from processing path", "type": "error_object"}

        return result_obj

    def _optimize_resources(self):
        """
        Optimize resource allocation based on current system state.
        Placeholder for actual implementation.
        """
        self.logger.info("Placeholder: Resource optimization triggered.")
        # current_allocation = self.resource_manager.check_resources()
        # optimal_allocation = self.resource_manager.optimize_resources()
        # if self._should_reallocate(current_allocation, optimal_allocation):
        #     self.resource_manager.apply_optimal_allocations(optimal_allocation)
        #     self.logger.info("Applied new resource allocation")

    def _update_overall_metrics(self, start_time_float: float, success: bool, reasoning_path_str: str,
                                query_id: Optional[str]):
        """
        Update overall system performance metrics.
        """
        try:
            response_time_seconds = time.time() - start_time_float
            self.performance_metrics['total_queries'] += 1
            if success:
                self.performance_metrics['successful_queries'] += 1
            else:
                self.performance_metrics['failed_queries'] += 1

            total_q = self.performance_metrics['total_queries']
            current_avg_rt = self.performance_metrics['avg_response_time']
            # Robust average calculation
            self.performance_metrics['avg_response_time'] = \
                ((current_avg_rt * (
                            total_q - 1)) + response_time_seconds) / total_q if total_q > 0 else response_time_seconds

            # Path performance (list of times per path) is updated in _update_path_specific_metrics

            # The complex reasoning_metrics update that was here is very specific to HotpotQA's
            # textual chain characteristics. For a more general SCM, it's better to let
            # MetricsCollector handle the details based on the data it receives.
            # If specific chain info (like length, confidence) is available from HybridIntegrator's result,
            # it could be passed to MetricsCollector.collect_query_metrics.

        except Exception as e:
            self.logger.error(f"Error updating overall metrics for qid {query_id}: {str(e)}")

    def _update_path_specific_metrics(self, path: str, duration_seconds: float, query_id: Optional[str]):
        """
        Update performance statistics for a specific reasoning path.
        """
        if path not in self.performance_metrics['path_performance']:
            self.performance_metrics['path_performance'][path] = []
        self.performance_metrics['path_performance'][path].append(duration_seconds)
        self.logger.debug(f"QID {query_id}: Path '{path}' took {duration_seconds:.4f}s. "
                          f"Count for this path: {len(self.performance_metrics['path_performance'][path])}, "
                          f"Avg time: {np.mean(self.performance_metrics['path_performance'][path]):.4f}s")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system performance metrics.
        """
        total_q = max(1, self.performance_metrics['total_queries'])  # Avoid division by zero

        success_rate = 0.0
        if self.performance_metrics['total_queries'] > 0:  # ensure total_queries is not zero
            success_rate = (self.performance_metrics['successful_queries'] / self.performance_metrics[
                'total_queries']) * 100

        error_rate = 0.0
        if self.performance_metrics['total_queries'] > 0:
            error_rate = (self.performance_metrics['failed_queries'] / self.performance_metrics['total_queries']) * 100

        metrics = {
            'total_queries': self.performance_metrics['total_queries'],
            'successful_queries': self.performance_metrics['successful_queries'],
            'failed_queries': self.performance_metrics['failed_queries'],
            'success_rate': success_rate,
            'error_rate': error_rate,
            'avg_response_time_seconds': self.performance_metrics['avg_response_time'],
            'path_distribution_percent': self._calculate_path_distribution_percentage(),
            'path_avg_times_seconds': {
                p: np.mean(times) if times else 0
                for p, times in self.performance_metrics['path_performance'].items()
            }
            # resource_efficiency is better calculated by MetricsCollector from detailed logs
        }
        return metrics

    def _calculate_path_distribution_percentage(self) -> Dict[str, float]:
        """
        Calculate the percentage distribution of reasoning paths taken.
        """
        total_recorded_paths = len(self.path_history)
        if not total_recorded_paths:
            return {}

        distribution_counts = defaultdict(int)
        for record in self.path_history:
            distribution_counts[record['path']] += 1

        return {
            path: (count / total_recorded_paths) * 100
            for path, count in distribution_counts.items()
        }

    def _calculate_resource_delta(
            self,
            initial_resources: Dict[str, float],
            final_resources: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate difference in resource usage, ensuring keys exist.
        """
        delta = {}
        all_keys = set(initial_resources.keys()) | set(final_resources.keys())
        for key in all_keys:
            delta[key] = final_resources.get(key, 0.0) - initial_resources.get(key, 0.0)
        return delta

    def _validate_resources(self, resource_status: Dict[str, float]) -> bool:
        """
        Check if resources are within acceptable limits based on configuration.
        """
        # Using thresholds from self.adaptive_thresholds for general pressure
        # More specific thresholds per resource could be in resource_manager's config
        if resource_status.get("cpu", 0.0) > self.adaptive_thresholds[
            'resource_pressure_threshold']:  # Example: 95% CPU
            self.logger.warning("High CPU usage detected.")
            return False
        if resource_status.get("memory", 0.0) > self.adaptive_thresholds[
            'resource_pressure_threshold']:  # Example: 95% Memory
            self.logger.warning("High Memory usage detected.")
            return False
        if self.resource_manager._gpu_available and resource_status.get("gpu", 0.0) > 0.99:  # Stricter for GPU
            self.logger.warning("High GPU usage detected.")
            return False
        return True

    def _should_reallocate(  # Not currently used, part of _optimize_resources placeholder
            self,
            current_allocation: Dict[str, float],
            optimal_allocation: Dict[str, float]
    ) -> bool:
        """
        Check if there's a significant difference to warrant reallocation.
        """
        threshold = 0.05  # Reallocate if difference is more than 5% for any resource
        for resource_key in current_allocation:
            if abs(optimal_allocation.get(resource_key, 0.0) - current_allocation[resource_key]) > threshold:
                return True
        return False

    def _handle_final_failure(self, start_time_float: float, reason_str: str, query_id: Optional[str],
                              is_error_object: bool = False) -> Dict[str, Any]:
        """
        Log failure and return a structured error response.
        """
        self.logger.error(f"Final failure for query_id {query_id} after retries or fatal error: {reason_str}")
        self._update_overall_metrics(start_time_float, False, "fallback_error",
                                     query_id)  # Mark as failure, path "fallback_error"

        error_payload: Any = reason_str
        if is_error_object:  # If the reason is already structured for DROP error
            if self.hybrid_integrator.dataset_type == 'drop':
                error_payload = {"error": reason_str, "type": "error_object",
                                 "spans": []}  # Ensure DROP eval can handle

        return {
            "query_id": query_id,
            "result": error_payload,  # For DROP, this might be the structured error object
            "error": reason_str,
            "status": "failed",
            "reasoning_path": "fallback_error",
            "processing_time": time.time() - start_time_float
        }

    def get_reasoning_path_stats(self) -> Dict[str, Any]:  # Used by main.py for summary
        """
        Get statistics on reasoning path distribution counts.
        """
        path_counts = defaultdict(lambda: {"count": 0, "total_time_seconds": 0.0})
        for record in self.path_history:
            path_counts[record['path']]["count"] += 1

        for path_name, time_list in self.performance_metrics['path_performance'].items():
            if time_list:
                path_counts[path_name]["total_time_seconds"] = sum(time_list)
                path_counts[path_name]["avg_time_seconds"] = np.mean(time_list)

        # Calculate percentage based on total_queries from overall metrics for consistency
        total_q_overall = self.performance_metrics['total_queries']

        final_path_stats = {}
        for path_name, data in path_counts.items():
            count = data["count"]
            final_path_stats[path_name] = {
                "count": count,
                "percentage": (count / total_q_overall) * 100 if total_q_overall > 0 else 0,
                "avg_time_seconds": data.get("avg_time_seconds", 0.0)
            }
        return final_path_stats