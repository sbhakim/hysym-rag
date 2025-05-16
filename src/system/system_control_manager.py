# src/system/system_control_manager.py

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
from datetime import datetime

# Ensure necessary imports from sibling directories are correct
# Assuming standard project structure where src is discoverable
try:
    from src.resources.resource_manager import ResourceManager
    from src.utils.metrics_collector import MetricsCollector
except ImportError:
    # Fallback if run directly or structure differs
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.resources.resource_manager import ResourceManager
    from src.utils.metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)

class UnifiedResponseAggregator:
    """
    Aggregates responses with optional explanations, supporting both HotpotQA (string) and DROP (dict) results.
    """
    def __init__(self, include_explanations: bool = False):
        self.include_explanations = include_explanations
        self.logger = logging.getLogger("UnifiedResponseAggregator")

    def aggregate(self, result: Any, source: str, confidence: float = 1.0, debug_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggregates the result with metadata, handling both HotpotQA and DROP result types.
        """
        output = {
            'result': result,
            'source': source,
            'confidence': confidence,
            'reasoning_path': source  # Keep reasoning_path consistent with source for now
        }
        # Ensure debug_info is checked before accessing keys within it
        if self.include_explanations and debug_info:
            output['debug_info'] = debug_info  # Store the whole debug info
            # Safely add specific keys if they exist in debug_info
            if 'fusion_strategy_text' in debug_info:
                output['fusion_strategy'] = debug_info['fusion_strategy_text']
            elif 'fusion_strategy_drop' in debug_info:
                output['fusion_strategy'] = debug_info['fusion_strategy_drop']
            # Add timings if present
            output['symbolic_time'] = debug_info.get('symbolic_time', 0.0)
            output['neural_time'] = debug_info.get('neural_time', 0.0)
            output['fusion_time'] = debug_info.get('fusion_time', 0.0)
        return output

    def format_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats the response dictionary for consistent output, handling different result types.
        """
        # Ensure 'result' key exists, providing a default if missing
        if 'result' not in data:
            self.logger.warning(f"Formatting response for QID {data.get('query_id', 'unknown')}: 'result' key missing. Setting default error.")
            data['result'] = data.get('error', "Error: No result generated.")  # Use provided error or a default

        result = data['result']

        # Handle DROP dataset specific formatting/validation
        if data.get('dataset_type') == 'drop':  # Assuming dataset_type might be passed in data
            if isinstance(result, dict):
                # Ensure essential DROP keys exist, provide defaults if not
                result.setdefault('number', '')
                result.setdefault('spans', [])
                result.setdefault('date', {'day': '', 'month': '', 'year': ''})
                # If an error key exists, ensure it's represented correctly
                if 'error' in result:
                     result.setdefault('type', 'error_object')  # Add type if error exists
            elif isinstance(result, str) and result.startswith("Error:"):
                 # If result is an error string, convert to standard DROP error format
                 error_msg = result
                 data['result'] = {'error': error_msg, 'type': 'error_object', 'spans': [], 'number': '', 'date': {'day':'','month':'','year':''}}
                 self.logger.warning(f"Converted error string to DROP dict for QID {data.get('query_id', 'unknown')}")
            else:
                # If it's not a dict or known error string, format is unexpected for DROP
                self.logger.warning(f"Unexpected result format for DROP QID {data.get('query_id', 'unknown')}: {result}. Wrapping as error.")
                error_msg = f"Unexpected result structure: {str(result)[:100]}"
                data['result'] = {'error': error_msg, 'type': 'error_object', 'spans': [], 'number': '', 'date': {'day':'','month':'','year':''}}

        # Handle Text QA (e.g., HotpotQA) - ensure result is a string
        elif not isinstance(result, str):
            self.logger.warning(f"Unexpected result type for Text QA QID {data.get('query_id', 'unknown')}: {type(result)}. Converting to string.")
            data['result'] = str(result)

        # Add explanation if requested and possible
        if self.include_explanations:
            # Ensure explanation is generated even if some keys are missing in 'data'
            data.setdefault("explanation", self._generate_detailed_explanation(data))

        # Ensure standard keys are present
        data.setdefault('query_id', 'unknown')
        data.setdefault('processing_time', 0.0)
        data.setdefault('resource_usage', {})
        data.setdefault('reasoning_path', 'unknown')
        data.setdefault('retries', 0)
        data.setdefault('status', 'unknown')  # e.g., 'success', 'failed'

        return data

    def _generate_detailed_explanation(self, data: Dict[str, Any]) -> str:
        """Generate detailed reasoning explanation string."""
        parts = []
        path = data.get('reasoning_path', 'unknown')
        parts.append(f"Reasoning Approach: {path}")

        proc_time = data.get('processing_time')
        if isinstance(proc_time, (int, float)):
             parts.append(f"Processing Time: {proc_time:.3f}s")

        resource_usage = data.get('resource_usage')
        if isinstance(resource_usage, dict):
            usage_parts = []
            for resource, usage in resource_usage.items():
                if isinstance(usage, (int, float)):
                    # Assume delta values don't need *100 anymore if calculated correctly before
                    usage_parts.append(f"{resource.capitalize()}: {usage:.2f} delta")
                else:
                    usage_parts.append(f"{resource.capitalize()}: {usage}")
            if usage_parts:
                 parts.append(f"Resource Delta: [{', '.join(usage_parts)}]")

        # Add component timings if available
        sym_time = data.get('symbolic_time')
        neu_time = data.get('neural_time')
        fus_time = data.get('fusion_time')
        timings = []
        if isinstance(sym_time, (int, float)) and sym_time > 0: timings.append(f"Sym: {sym_time:.3f}s")
        if isinstance(neu_time, (int, float)) and neu_time > 0: timings.append(f"Neu: {neu_time:.3f}s")
        if isinstance(fus_time, (int, float)) and fus_time > 0: timings.append(f"Fus: {fus_time:.3f}s")
        if timings:
             parts.append(f"Component Times: [{', '.join(timings)}]")

        if data.get('retries', 0) > 0:
            parts.append(f"Retries Attempted: {data['retries']}")

        status = data.get('status', 'unknown')
        if status == 'failed':
             parts.append(f"Status: Failed ({data.get('error', 'Unknown reason')})")

        return " | ".join(parts) if parts else "No detailed explanation available."


class SystemControlManager:
    """
    Orchestrates the hybrid reasoning pipeline with resource management and fallback strategies.
    Supports both HotpotQA and DROP datasets through dataset_type parameter.
    """

    def __init__(
            self,
            hybrid_integrator,
            resource_manager: ResourceManager,
            aggregator: UnifiedResponseAggregator,
            metrics_collector: MetricsCollector,
            error_retry_limit: int = 2,
            max_query_time: float = 30.0,
    ):
        if not all([hybrid_integrator, resource_manager, aggregator, metrics_collector]):
            raise ValueError("All core components (integrator, resource_manager, aggregator, metrics_collector) must be provided.")

        self.hybrid_integrator = hybrid_integrator
        self.resource_manager = resource_manager
        self.aggregator = aggregator
        self.metrics_collector = metrics_collector
        self.error_retry_limit = max(error_retry_limit, 0)  # Ensure non-negative
        self.max_query_time = max(max_query_time, 1.0)  # Ensure positive time limit
        self.logger = logger
        self.logger.setLevel(logging.INFO)  # Set default level

        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'error_count': 0,
            'path_usage': defaultdict(int)  # Tracks how often each path is *chosen*
        }
        # Stores counts, success counts, and average time per path type (updated upon completion)
        self.reasoning_path_stats = defaultdict(lambda: {'count': 0, 'success': 0, 'total_time': 0.0, 'avg_time': 0.0})
        # Stores details about each path selection decision made
        self.path_history = []
        # Dynamic path selection thresholds (will be adjusted by _optimize_resources)
        self.low_complexity_thr = 0.4
        self.high_complexity_thr = 0.8
        self.low_resource_thr = 0.6
        self.high_resource_thr = 0.85

    def process_query_with_fallback(
            self,
            query: str,
            context: str,
            query_id: str,
            forced_path: Optional[str] = None,
            query_complexity: float = 0.5,
            supporting_facts: Optional[List[Tuple[str, int]]] = None,
            dataset_type: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Processes a query with fallback strategies in case of errors.
        Now accepts dataset_type for path selection.
        Returns a tuple containing:
            - A dictionary representing the final formatted response or error details.
            - A string indicating the final reasoning path taken ('symbolic', 'neural', 'hybrid', 'fallback_error', etc.).
        """
        self.logger.info(f"Processing query_id: {query_id}, Query: '{query[:50]}...', Dataset: {dataset_type or 'unknown'}")
        self.performance_metrics['total_queries'] += 1
        attempts = 0
        max_retries = self.error_retry_limit
        reasoning_path_taken = 'unknown'  # Will be updated after path selection

        # Record initial resource state for delta calculation
        initial_resources = self.resource_manager.check_resources()
        overall_start_time = time.time()  # Track start time for overall processing, including retries

        while attempts <= max_retries:
            attempt_start_time = time.time()  # Start timer for this specific attempt
            try:
                # Optimize resources before processing (adjust thresholds if needed)
                self._optimize_resources()

                # Select path and execute the query processing along that path
                final_result_obj, reasoning_path_taken = self._timed_process_query(
                    query,
                    context,
                    forced_path,
                    query_complexity,
                    supporting_facts,
                    query_id,
                    dataset_type
                )

                # Calculate processing time for this successful attempt
                processing_time_seconds = time.time() - attempt_start_time

                # Check if the result object indicates an internal error occurred
                is_error_result = False
                if isinstance(final_result_obj, dict) and (final_result_obj.get("type") == "error_object" or final_result_obj.get("status") == "error"):
                    is_error_result = True
                    self.logger.error(f"QID {query_id} Path {reasoning_path_taken} resulted in error object: {final_result_obj.get('error', 'Unknown internal error')}")
                elif isinstance(final_result_obj, str) and final_result_obj.startswith("Error:"):
                    is_error_result = True
                    self.logger.error(f"QID {query_id} Path {reasoning_path_taken} resulted in error string: {final_result_obj}")

                # If an error occurred *within* the successful execution of a path
                if is_error_result:
                    # Still counts as an attempt, log error, maybe retry differently?
                    # For now, let's treat it like other exceptions for retry logic
                    raise ValueError(f"Internal error returned from path {reasoning_path_taken}: {final_result_obj}")

                # --- Success Case ---
                # Calculate resource delta based on overall start
                final_utilized_resources = self.resource_manager.check_resources()
                resource_delta = {k: final_utilized_resources[k] - initial_resources.get(k, 0.0)
                                  for k in final_utilized_resources if k != 'timestamp'}

                # Update Success Stats
                self.performance_metrics['successful_queries'] += 1
                self.performance_metrics['path_usage'][reasoning_path_taken] += 1  # Track chosen path usage
                self._update_reasoning_path_stats(reasoning_path_taken, success=True, time_taken=processing_time_seconds)

                # Collect academic metrics
                prediction_for_metrics = str(final_result_obj)  # Default string representation
                if dataset_type == 'drop' and isinstance(final_result_obj, dict):
                    # Create specific string representations for DROP types for logging/metrics
                    num = final_result_obj.get('number')
                    spans = final_result_obj.get('spans')
                    date = final_result_obj.get('date')
                    error = final_result_obj.get('error')
                    if num: prediction_for_metrics = f"DROP_Num: {num}"
                    elif spans: prediction_for_metrics = f"DROP_Spans: {str(spans)[:50]}"
                    elif date and any(date.values()): prediction_for_metrics = f"DROP_Date: {date.get('month','')}/{date.get('day','')}/{date.get('year','')}"
                    elif error: prediction_for_metrics = f"DROP_Error: {error}"
                    else: prediction_for_metrics = "DROP_Empty"

                # Collect metrics using the metrics collector
                if hasattr(self, 'metrics_collector') and self.metrics_collector:
                     self.metrics_collector.collect_query_metrics(
                         query=query,
                         prediction=prediction_for_metrics,
                         ground_truth=None,  # Compared later
                         reasoning_path=reasoning_path_taken,
                         processing_time=processing_time_seconds,
                         resource_usage=resource_delta,
                         complexity_score=query_complexity,
                         query_id=query_id
                     )

                # Format the final response structure using the aggregator
                # Add component timings extracted from the result object if they exist
                component_timings = {}
                if isinstance(final_result_obj, dict):
                    component_timings = {
                        k: final_result_obj.get(k) for k in ['symbolic_time', 'neural_time', 'fusion_time']
                        if k in final_result_obj and isinstance(final_result_obj.get(k), (int, float))
                    }

                formatted_response = self.aggregator.format_response({
                    'query_id': query_id,
                    'result': final_result_obj,
                    'processing_time': processing_time_seconds,
                    'resource_usage': resource_delta,  # Store the calculated delta
                    'reasoning_path': reasoning_path_taken,
                    'retries': attempts,
                    'status': 'success',
                    'dataset_type': dataset_type,  # Include dataset type
                    **component_timings  # Merge timings if they exist
                })

                return formatted_response, reasoning_path_taken  # Return success dict and path string

            except Exception as e:
                # --- Error during attempt ---
                self.logger.exception(f"Error processing query_id {query_id} (attempt {attempts + 1}/{max_retries + 1}): {str(e)}")
                attempts += 1
                self.performance_metrics['error_count'] += 1  # Increment error count here
                if attempts > max_retries:
                    # Final failure after all retries
                    # Pass the *overall* start time for accurate duration logging
                    failure_dict, failure_path = self._handle_final_failure(
                        overall_start_time, str(e), query_id, dataset_type
                    )
                    return failure_dict, failure_path  # Return error dict and 'fallback_error' path
                # Log retry attempt
                self.logger.info(f"Retrying query_id {query_id} ({attempts}/{max_retries})...")
                time.sleep(0.5 * attempts)  # Exponential backoff before next attempt
                # Continue to the next iteration of the while loop to retry
                continue

        # Should not be reached if logic is correct, but acts as a final fallback
        self.logger.error(f"Exited retry loop unexpectedly for QID {query_id}")
        return self._handle_final_failure(overall_start_time, "Exited retry loop unexpectedly", query_id, dataset_type)

    def _timed_process_query(
            self,
            query: str,
            context: str,
            forced_path: Optional[str],
            query_complexity: float,
            supporting_facts: Optional[List[Tuple[str, int]]],
            query_id: str,
            dataset_type: Optional[str]
    ) -> Tuple[Any, str]:
        """
        Selects the processing path and executes the query along that path with fallbacks.
        Returns the raw result object (string or dict) and the actual path taken.
        Handles exceptions during execution with fallback paths for DROP.
        """
        self.logger.debug(f"Executing query for QID {query_id} with dataset_type '{dataset_type}'...")
        dt_lower = dataset_type.lower() if dataset_type else 'unknown'
        attempted_paths = []

        # Initial path selection
        path_selection_start_time = time.time()
        initial_path = forced_path if forced_path else self.select_reasoning_path(query_complexity, query_id, dataset_type)
        self.logger.debug(f"Path selection for QID {query_id} took: {time.time() - path_selection_start_time:.4f}s. Initial Path: {initial_path}")

        # List of paths to try in order (starting with the selected path)
        paths_to_try = [initial_path]
        if dt_lower == 'drop' and not forced_path:
            # For DROP, define fallback paths if the initial path fails
            if initial_path == 'hybrid':
                paths_to_try.extend(['symbolic', 'neural'])
            elif initial_path == 'symbolic':
                paths_to_try.extend(['hybrid', 'neural'])
            elif initial_path == 'neural':
                paths_to_try.extend(['hybrid', 'symbolic'])

        for path in paths_to_try:
            attempted_paths.append(path)
            self.logger.debug(f"[DROP QID:{query_id}] Attempting path '{path}' (Attempted: {attempted_paths})")
            try:
                result_obj = self._execute_processing_path(
                    path, query, context, query_complexity, supporting_facts, query_id, dataset_type
                )
                self.logger.info(f"[DROP QID:{query_id}] Successfully executed path '{path}'")
                return result_obj, path
            except Exception as e:
                self.logger.warning(f"[DROP QID:{query_id}] Path '{path}' failed: {str(e)}. Trying next path...")
                continue

        # If all paths fail, raise the last exception to trigger retry logic in process_query_with_fallback
        self.logger.error(f"[DROP QID:{query_id}] All paths failed: {attempted_paths}")
        raise ValueError(f"All paths failed for QID {query_id}: {attempted_paths}")

    def select_reasoning_path(
            self,
            query_complexity: float,
            query_id: str,
            dataset_type: Optional[str]
    ) -> str:
        """
        Selects reasoning path based on query complexity, resource availability, and dataset type.
        Logs the decision factors and outcome.
        Enhanced for DROP to consider query complexity more granularly.
        """
        resource_metrics = self.resource_manager.check_resources()
        # Handle potential None values from resource check if GPU is unavailable
        cpu_usage = resource_metrics.get('cpu', 0.0) or 0.0
        memory_usage = resource_metrics.get('memory', 0.0) or 0.0
        gpu_usage = resource_metrics.get('gpu', 0.0) or 0.0
        overall_resource_pressure = max(cpu_usage, memory_usage, gpu_usage)

        decision_factors = {
            'query_complexity': round(query_complexity, 3),
            'cpu_usage': round(cpu_usage, 3),
            'memory_usage': round(memory_usage, 3),
            'gpu_usage': round(gpu_usage, 3),
            'overall_resource_pressure': round(overall_resource_pressure, 3),
            'dataset_type': dataset_type or 'unknown'
        }
        self.logger.info(f"Path Selection Factors for QID {query_id}: {decision_factors}")

        # --- Path Selection Logic ---
        path = "hybrid"
        reason = "Default choice: suitable for balanced complexity/resources or DROP dataset."

        dt_lower = dataset_type.lower() if dataset_type else 'unknown'

        if dt_lower == 'drop':
            # Enhanced logic for DROP queries
            if query_complexity < self.low_complexity_thr and overall_resource_pressure < self.low_resource_thr:
                path = "symbolic"
                reason = f"DROP dataset: Low complexity (<{self.low_complexity_thr}) and low resources (<{self.low_resource_thr}) favor symbolic."
            elif query_complexity > self.high_complexity_thr or overall_resource_pressure > self.high_resource_thr:
                path = "neural"
                if query_complexity > self.high_complexity_thr:
                    reason = f"DROP dataset: High query complexity (> {self.high_complexity_thr}) favors neural."
                else:
                    reason = f"DROP dataset: High resource pressure (> {self.high_resource_thr}) favors neural."
            else:
                path = "hybrid"
                reason = f"DROP dataset: Balanced complexity ({query_complexity}) and resources ({overall_resource_pressure}) favor hybrid."
        else:
            # Logic for text-based QA (e.g., HotpotQA)
            if query_complexity < self.low_complexity_thr and overall_resource_pressure < self.low_resource_thr:
                path = "symbolic"
                reason = f"Low complexity (<{self.low_complexity_thr}) and low resources (<{self.low_resource_thr}) favor symbolic."
            elif query_complexity >= self.high_complexity_thr or overall_resource_pressure >= self.high_resource_thr:
                path = "neural"
                if query_complexity >= self.high_complexity_thr:
                    reason = f"High query complexity (>= {self.high_complexity_thr}) favors neural."
                else:
                    reason = f"High resource pressure (>= {self.high_resource_thr}) favors neural."

        # --- Logging Decision ---
        self.logger.info(f"Path Selection Decision for QID {query_id}: Chosen Path='{path}' (Reason: {reason})")

        # Append decision details to path_history list
        try:
            self.path_history.append({
                'timestamp': datetime.now().isoformat(),
                'query_id': query_id,
                'chosen_path': path,
                'factors': decision_factors,
                'reason': reason
            })
        except Exception as hist_err:
            self.logger.error(f"Failed to append to path_history for QID {query_id}: {hist_err}")

        return path

    def _optimize_resources(self):
        """
        Implements resource optimization logic by analyzing resource usage trends and path performance.
        Adjusts path selection thresholds dynamically.
        """
        try:
            # Analyze recent resource usage trends
            resource_metrics = self.resource_manager.check_resources()
            cpu_usage = resource_metrics.get('cpu', 0.0) or 0.0
            memory_usage = resource_metrics.get('memory', 0.0) or 0.0
            gpu_usage = resource_metrics.get('gpu', 0.0) or 0.0
            overall_pressure = max(cpu_usage, memory_usage, gpu_usage)

            # Analyze path performance from reasoning_path_stats
            path_performance = {}
            for path, stats in self.reasoning_path_stats.items():
                count = stats.get('count', 0)
                if count > 0:
                    success_rate = (stats.get('success', 0) / count) * 100
                    avg_time = stats.get('avg_time', 0.0)
                    path_performance[path] = {'success_rate': success_rate, 'avg_time': avg_time}
                else:
                    path_performance[path] = {'success_rate': 0.0, 'avg_time': 0.0}

            self.logger.debug(f"Resource Optimization: Current Usage - CPU: {cpu_usage:.2f}, Memory: {memory_usage:.2f}, GPU: {gpu_usage:.2f}, Overall Pressure: {overall_pressure:.2f}")
            self.logger.debug(f"Path Performance: {path_performance}")

            # Adjust thresholds based on resource usage and path performance
            adjustments = []
            if overall_pressure > 0.9:
                # High resource pressure: make symbolic path more likely
                if self.low_complexity_thr < 0.6:
                    self.low_complexity_thr += 0.05
                    adjustments.append(f"Increased low_complexity_thr to {self.low_complexity_thr:.2f} due to high resource pressure")
                if self.high_complexity_thr > 0.6:
                    self.high_complexity_thr -= 0.05
                    adjustments.append(f"Decreased high_complexity_thr to {self.high_complexity_thr:.2f} due to high resource pressure")
            elif overall_pressure < 0.3:
                # Low resource pressure: make neural path more likely
                if self.low_complexity_thr > 0.2:
                    self.low_complexity_thr -= 0.05
                    adjustments.append(f"Decreased low_complexity_thr to {self.low_complexity_thr:.2f} due to low resource pressure")
                if self.high_complexity_thr < 0.9:
                    self.high_complexity_thr += 0.05
                    adjustments.append(f"Increased high_complexity_thr to {self.high_complexity_thr:.2f} due to low resource pressure")

            # Adjust based on path performance
            if 'neural' in path_performance and path_performance['neural']['success_rate'] < 50 and path_performance['neural']['avg_time'] > 1.0:
                # Neural path is underperforming and slow, shift towards symbolic/hybrid
                if self.high_complexity_thr > 0.6:
                    self.high_complexity_thr -= 0.05
                    adjustments.append(f"Decreased high_complexity_thr to {self.high_complexity_thr:.2f} due to poor neural performance")
            if 'symbolic' in path_performance and path_performance['symbolic']['success_rate'] < 50 and path_performance['symbolic']['avg_time'] > 1.0:
                # Symbolic path is underperforming and slow, shift towards neural/hybrid
                if self.low_complexity_thr > 0.2:
                    self.low_complexity_thr -= 0.05
                    adjustments.append(f"Decreased low_complexity_thr to {self.low_complexity_thr:.2f} due to poor symbolic performance")

            if adjustments:
                self.logger.info(f"Resource Optimization Adjustments: {', '.join(adjustments)}")
            else:
                self.logger.info("Resource Optimization: No adjustments needed at this time.")

        except Exception as e:
            self.logger.error(f"Error in resource optimization: {str(e)}")
            # Revert to default thresholds if optimization fails
            self.low_complexity_thr = 0.4
            self.high_complexity_thr = 0.8
            self.low_resource_thr = 0.6
            self.high_resource_thr = 0.85
            self.logger.info("Reverted to default thresholds due to optimization error.")

    def _execute_processing_path(
            self,
            path: str,
            query: str,
            context: str,
            query_complexity: float,
            supporting_facts: Optional[List[Tuple[str, int]]],
            query_id: str,
            dataset_type: Optional[str]
    ) -> Any:
        """
        Executes the query processing along the specified path.
        Returns the raw result object (string or dict) or raises an exception on failure.
        Component timings are now primarily handled within the HybridIntegrator if possible.
        """
        self.logger.debug(f"Executing path '{path}' for QID {query_id}...")
        result_obj: Any = None

        try:
            if path == "symbolic":
                if not hasattr(self.hybrid_integrator, 'symbolic_reasoner') or not self.hybrid_integrator.symbolic_reasoner:
                     raise RuntimeError("Symbolic reasoner component is not available.")
                result_obj = self.hybrid_integrator.symbolic_reasoner.process_query(
                    query, context=context, dataset_type=dataset_type, query_id=query_id
                )
            elif path == "neural":
                 if not hasattr(self.hybrid_integrator, 'neural_retriever') or not self.hybrid_integrator.neural_retriever:
                      raise RuntimeError("Neural retriever component is not available.")
                 # Assuming symbolic guidance isn't available/needed for a pure neural path
                 result_obj = self.hybrid_integrator.neural_retriever.retrieve_answer(
                     context, query, symbolic_guidance=None, query_complexity=query_complexity, dataset_type=dataset_type
                 )
            elif path == "hybrid":
                 if not self.hybrid_integrator:
                      raise RuntimeError("Hybrid integrator component is not available.")
                 # process_query returns tuple (result, source), we need the result
                 result_obj, _ = self.hybrid_integrator.process_query(
                     query, context, query_complexity=query_complexity, supporting_facts=supporting_facts, query_id=query_id
                 )
            else:
                # This case should ideally be prevented by path selection validation
                self.logger.error(f"Attempted to execute unknown path '{path}' for QID {query_id}.")
                raise ValueError(f"Unknown processing path: {path}")

            self.logger.debug(f"Path '{path}' execution completed for QID {query_id}.")
            # Basic validation of the result before returning
            if result_obj is None:
                 self.logger.warning(f"Path '{path}' execution for QID {query_id} returned None.")
                 # Return a standard error format based on dataset type
                 error_msg = f"Error: Path '{path}' returned no result."
                 if dataset_type == 'drop':
                      return {"error": error_msg, "type": "error_object", "spans": [], 'number': '', 'date': {'day':'','month':'','year':''}}
                 else:
                      return error_msg

            return result_obj  # Return the result (string, dict, or error object)

        except Exception as e:
            # Log the exception details and re-raise it
            # The retry logic in process_query_with_fallback will catch this
            self.logger.exception(f"Critical exception during path execution '{path}' for QID {query_id}: {str(e)}")
            raise e  # Re-raise the original exception

    def _update_reasoning_path_stats(self, path: str, success: bool, time_taken: Optional[float] = None):
        """Update statistics for a specific reasoning path upon completion."""
        if not path or path == 'unknown':  # Don't track 'unknown' path stats
             return
        # Ensure path entry exists using defaultdict behavior
        stats = self.reasoning_path_stats[path]
        stats['count'] += 1
        if success:
            stats['success'] += 1
        if time_taken is not None:
             # Ensure time_taken is a valid number
             if isinstance(time_taken, (int, float)) and time_taken >= 0:
                  stats['total_time'] += time_taken
                  # Calculate running average safely
                  stats['avg_time'] = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0.0
             else:
                  self.logger.warning(f"Invalid time_taken value ({time_taken}) received for path '{path}'. Ignoring for average calculation.")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns a summary of overall system performance metrics."""
        total_q = self.performance_metrics.get('total_queries', 0)
        successful_queries = self.performance_metrics.get('successful_queries', 0)

        # Calculate overall average response time from successful path stats
        total_time_all_successful = sum(stats.get('total_time', 0.0) for path, stats in self.reasoning_path_stats.items() if path != 'fallback_error')
        total_successful_runs = successful_queries  # Use the successful query count

        avg_resp_time = (total_time_all_successful / total_successful_runs) if total_successful_runs > 0 else 0.0

        metrics = {
            'total_queries': total_q,
            'successful_queries': successful_queries,
            'success_rate': (successful_queries / total_q) * 100 if total_q > 0 else 0.0,
            'error_count': self.performance_metrics.get('error_count', 0),
            'avg_successful_response_time_sec': avg_resp_time,
            'path_chosen_distribution': dict(self.performance_metrics.get('path_usage', defaultdict(int)))
        }
        return metrics

    def get_reasoning_path_stats(self) -> Dict[str, Any]:
        """Returns detailed statistics for each reasoning path executed."""
        total_q = self.performance_metrics.get('total_queries', 0)
        if total_q == 0: total_q = 1  # Avoid division by zero if no queries processed

        stats = {}
        for path, data in self.reasoning_path_stats.items():
            count = data.get('count', 0)
            success = data.get('success', 0)
            avg_time = data.get('avg_time', 0.0)
            # Calculate success rate for this specific path
            path_success_rate = (success / count) * 100 if count > 0 else 0.0
            stats[path] = {
                'execution_count': count,
                'success_count': success,
                'path_success_rate_percent': round(path_success_rate, 2),
                'avg_time_sec': round(avg_time, 3),
                'percentage_of_total_queries': round((count / total_q) * 100, 2)
            }
        return stats

    def _handle_final_failure(self, overall_start_time: float, reason_str: str, query_id: str, dataset_type: Optional[str]) -> Tuple[Dict[str, Any], str]:
        """Logs final failure, updates stats, and returns a structured error response tuple (dict, str)."""
        self.logger.error(f"Final failure processing QID {query_id} after retries or fatal error: {reason_str}")
        processing_time = time.time() - overall_start_time
        error_path_name = 'fallback_error'

        # Create dataset-specific error structure for the 'result' field
        error_result_payload: Union[str, Dict]
        if dataset_type == 'drop':
            error_result_payload = {"error": reason_str, "type": "error_object", "spans": [], 'number': '', 'date': {'day':'','month':'','year':''}}
        else:
            error_result_payload = f"Error: {reason_str}"  # Simple error string for text QA

        # Update stats for the 'fallback_error' path
        self._update_reasoning_path_stats(error_path_name, success=False, time_taken=processing_time)

        # Construct the final error response dictionary to be returned
        error_response_dict = {
            'query_id': query_id,
            'result': error_result_payload,  # Contains the error string or dict
            'error': reason_str,  # Explicit error reason string
            'status': 'failed',
            'reasoning_path': error_path_name,
            'processing_time': processing_time,
            'retries': self.error_retry_limit  # Indicates max retries were used
        }

        # Format using aggregator to ensure consistent structure
        formatted_error_response = self.aggregator.format_response(error_response_dict)

        # Return the formatted error dictionary and the standard error path name
        return formatted_error_response, error_path_name