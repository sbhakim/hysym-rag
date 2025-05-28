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
    # UnifiedResponseAggregator will be passed as an instance to the constructor.
    # It's now defined in src.system.response_aggregator.py and imported via src.system.__init__.py
    from .system_logic_helpers import _determine_reasoning_path_logic, _optimize_thresholds_logic
    from src.reasoners.dummy_reasoners import DummySymbolicReasoner, DummyNeuralRetriever # <<<--- ADD THIS LINE
except ImportError:
    # Fallback if run directly or structure differsfv
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.resources.resource_manager import ResourceManager
    from src.utils.metrics_collector import MetricsCollector
    from src.system.system_logic_helpers import _determine_reasoning_path_logic, _optimize_thresholds_logic
    from src.reasoners.dummy_reasoners import DummySymbolicReasoner, DummyNeuralRetriever

logger = logging.getLogger(__name__)


# The UnifiedResponseAggregator class definition has been moved to src/system/response_aggregator.py

class SystemControlManager:
    """
    Orchestrates the hybrid reasoning pipeline with resource management and fallback strategies.
    Supports both HotpotQA and DROP datasets through dataset_type parameter.
    """

    def __init__(
            self,
            hybrid_integrator: Any,  # Assuming HybridIntegrator type
            resource_manager: ResourceManager,
            aggregator: Any,  # This will be an instance of UnifiedResponseAggregator
            metrics_collector: MetricsCollector,
            error_retry_limit: int = 2,
            max_query_time: float = 30.0,
            use_adaptive_logic: bool = True  # << NEW PARAMETER
    ):
        if not all([hybrid_integrator, resource_manager, aggregator, metrics_collector]):
            raise ValueError(
                "All core components (integrator, resource_manager, aggregator, metrics_collector) must be provided.")

        self.hybrid_integrator = hybrid_integrator
        self.resource_manager = resource_manager
        self.aggregator = aggregator  # Instance of UnifiedResponseAggregator
        self.metrics_collector = metrics_collector
        self.error_retry_limit = max(error_retry_limit, 0)
        self.max_query_time = max(max_query_time, 1.0)
        self.logger = logger
        self.logger.setLevel(logging.INFO)
        self.use_adaptive_logic = use_adaptive_logic  # << STORE THE FLAG

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
        # Dynamic path selection thresholds
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
        Returns a tuple containing the final formatted response and the reasoning path taken.
        """
        self.logger.info(
            f"Processing query_id: {query_id}, Query: '{query[:50]}...', Dataset: {dataset_type or 'unknown'}")
        self.performance_metrics['total_queries'] += 1
        attempts = 0
        # Ensure max_retries is non-negative; self.error_retry_limit is already max(val, 0) in __init__
        max_retries = self.error_retry_limit
        reasoning_path_taken = 'unknown_initial'  # More specific initial value

        initial_resources = self.resource_manager.check_resources()
        overall_start_time = time.time()
        # Store the full result object from the last successful or failed attempt within the loop
        last_attempt_result_obj: Optional[Any] = None

        while attempts <= max_retries:
            attempt_start_time = time.time()
            current_attempt_final_result_obj: Optional[Any] = None  # Result of this specific attempt
            current_attempt_reasoning_path: str = 'unknown_this_attempt'

            try:
                self._optimize_resources()  # Ensure this is lightweight or conditional

                # _timed_process_query returns (raw_result_from_integrator_or_path, path_string)
                current_attempt_final_result_obj, current_attempt_reasoning_path = self._timed_process_query(
                    query, context, forced_path, query_complexity,
                    supporting_facts, query_id, dataset_type
                )
                reasoning_path_taken = current_attempt_reasoning_path  # Update path taken for this successful/failed attempt
                last_attempt_result_obj = current_attempt_final_result_obj  # Store this attempt's result

                processing_time_seconds = time.time() - attempt_start_time

                is_error_result = False
                # Check if the raw result object indicates an error
                if isinstance(current_attempt_final_result_obj, dict) and \
                        (current_attempt_final_result_obj.get("type") == "error_object" or \
                         current_attempt_final_result_obj.get("status") == "error" or \
                         current_attempt_final_result_obj.get("error") is not None):  # Added check for 'error' key
                    is_error_result = True
                    error_message = current_attempt_final_result_obj.get('error',
                                                                         current_attempt_final_result_obj.get(
                                                                             'rationale',
                                                                             'Unknown internal error'))
                    self.logger.error(
                        f"QID {query_id} Path {reasoning_path_taken} resulted in error object: {error_message}")
                elif isinstance(current_attempt_final_result_obj, str) and current_attempt_final_result_obj.startswith(
                        "Error:"):
                    is_error_result = True
                    error_message = current_attempt_final_result_obj
                    self.logger.error(
                        f"QID {query_id} Path {reasoning_path_taken} resulted in error string: {error_message}")

                if is_error_result:
                    # Pass the specific error message/object from this attempt
                    raise ValueError(f"Internal error returned from path {reasoning_path_taken}: {error_message}")

                # --- Success Case for this attempt ---
                final_utilized_resources = self.resource_manager.check_resources()
                resource_delta = {k: final_utilized_resources.get(k, 0.0) - initial_resources.get(k, 0.0)
                                  for k in final_utilized_resources if k != 'timestamp'}

                # Increment successful_queries only once per query_id if it ultimately succeeds.
                # This is handled by checking if the query_id is already marked as successful in path_history.
                # A simpler approach: if a query succeeds on any attempt, it's one successful query.
                # The current logic in get_performance_metrics counts based on self.performance_metrics['successful_queries']
                # which we will increment here.
                if not any(h['query_id'] == query_id and h.get('final_status') == 'success' for h in self.path_history):
                    self.performance_metrics['successful_queries'] += 1
                    # Log this success to path_history to prevent recount on retry success (if that logic is added)
                    # This might need adjustment if retries can improve an already 'successful' but poor answer.
                    # For now, first success counts.
                    if hasattr(self, 'path_history'):  # Ensure path_history exists
                        self.path_history.append({'query_id': query_id, 'final_status': 'success', 'attempt': attempts})

                self.performance_metrics['path_usage'][reasoning_path_taken] += 1
                self._update_reasoning_path_stats(reasoning_path_taken, success=True,
                                                  time_taken=processing_time_seconds)

                # --- MODIFICATION: Pass the RAW result object to MetricsCollector ---
                # current_attempt_final_result_obj is the raw output (dict for DROP, str for Text QA)
                # from the hybrid_integrator or reasoner. This is what _calculate_performance_metrics expects for DROP.
                if hasattr(self, 'metrics_collector') and self.metrics_collector:
                    self.metrics_collector.collect_query_metrics(
                        query=query,
                        prediction=current_attempt_final_result_obj,  # Pass the raw object
                        ground_truth=None,  # Ground truth is evaluated in main.py after this call
                        reasoning_path=reasoning_path_taken,
                        processing_time=processing_time_seconds,  # Time for this successful attempt
                        resource_usage=resource_delta,
                        complexity_score=query_complexity,
                        query_id=query_id,
                        confidence=current_attempt_final_result_obj.get('confidence') if isinstance(
                            current_attempt_final_result_obj, dict) else None,
                        operation_type=current_attempt_final_result_obj.get('type') if isinstance(
                            current_attempt_final_result_obj, dict) else None
                    )

                component_timings = {}
                if isinstance(current_attempt_final_result_obj, dict):
                    component_timings['symbolic_time'] = current_attempt_final_result_obj.get('symbolic_time', 0.0)
                    component_timings['neural_time'] = current_attempt_final_result_obj.get('neural_time', 0.0)
                    component_timings['fusion_time'] = current_attempt_final_result_obj.get('fusion_time', 0.0)

                # The aggregator.format_response will create the final dict for the caller (main.py)
                formatted_response = self.aggregator.format_response({
                    'query_id': query_id,
                    'result': current_attempt_final_result_obj,  # Pass the raw result to aggregator
                    'processing_time': processing_time_seconds,
                    'resource_usage': resource_delta,
                    'reasoning_path': reasoning_path_taken,
                    'retries': attempts,
                    'status': 'success',
                    'dataset_type': dataset_type,
                    **component_timings
                })
                return formatted_response, reasoning_path_taken

            except Exception as e:
                # This block handles errors from _timed_process_query or the ValueError raised above
                attempt_processing_time = time.time() - attempt_start_time
                self.logger.exception(
                    f"Error processing query_id {query_id} (attempt {attempts + 1}/{max_retries + 1}) with path '{reasoning_path_taken}': {str(e)}")
                self.performance_metrics['error_count'] += 1  # Count each failed attempt

                # --- MODIFICATION: Log failed attempt to MetricsCollector ---
                # last_attempt_result_obj should hold the error object from the failed path if one was returned,
                # or we construct one.
                error_payload_for_collector: Any
                if isinstance(last_attempt_result_obj, dict) and (
                        last_attempt_result_obj.get('error') or last_attempt_result_obj.get('status') == 'error'):
                    error_payload_for_collector = last_attempt_result_obj
                else:  # Construct a generic error object if last_attempt_result_obj is not a structured error
                    error_message_str = str(e)
                    if dataset_type == 'drop':
                        error_payload_for_collector = {
                            'number': '', 'spans': [], 'date': {'day': '', 'month': '', 'year': ''},
                            'status': 'error', 'confidence': 0.0,
                            'rationale': f"Attempt {attempts + 1} failed on path {reasoning_path_taken}: {error_message_str}",
                            'type': 'error_attempt_failure', 'error': error_message_str
                        }
                    else:  # Text QA
                        error_payload_for_collector = f"Error on attempt {attempts + 1} (path {reasoning_path_taken}): {error_message_str}"

                current_resources_on_fail = self.resource_manager.check_resources()
                resource_delta_on_fail = {
                    k: current_resources_on_fail.get(k, 0.0) - initial_resources.get(k, 0.0)
                    for k in current_resources_on_fail if k != 'timestamp'
                }

                if hasattr(self, 'metrics_collector') and self.metrics_collector:
                    self.metrics_collector.collect_query_metrics(
                        query=query,
                        prediction=error_payload_for_collector,  # Pass the error object/string
                        ground_truth=None,
                        reasoning_path=reasoning_path_taken if reasoning_path_taken != 'unknown_initial' else "path_unknown_during_error",
                        processing_time=attempt_processing_time,
                        resource_usage=resource_delta_on_fail,
                        complexity_score=query_complexity,
                        query_id=query_id,
                        confidence=error_payload_for_collector.get('confidence', 0.0) if isinstance(
                            error_payload_for_collector, dict) else 0.0,
                        operation_type=error_payload_for_collector.get('type', 'error_attempt_failure') if isinstance(
                            error_payload_for_collector, dict) else 'error_attempt_failure'
                    )
                # --- END MODIFICATION ---

                attempts += 1
                # Update stats for the failed path attempt
                self._update_reasoning_path_stats(
                    reasoning_path_taken if reasoning_path_taken != 'unknown_initial' else "attempt_failed_path_unknown",
                    success=False,
                    time_taken=attempt_processing_time
                )

                if attempts > max_retries:
                    # Pass the reason for the final failure (the last exception encountered)
                    return self._handle_final_failure(
                        overall_start_time, str(e), query_id, dataset_type,
                        reasoning_path_taken if reasoning_path_taken != 'unknown_initial' else "path_unknown_at_final_failure"
                    )
                self.logger.info(f"Retrying query_id {query_id} (attempt {attempts}/{max_retries})...")
                # Basic backoff, consider more sophisticated strategies if needed
                time.sleep(min(0.5 * attempts, 2.0))  # Sleep up to 2 seconds
                # Reset initial_resources for the next attempt to measure its delta correctly
                initial_resources = self.resource_manager.check_resources()
                continue  # To the next iteration of the while loop

        # Fallback if loop finishes unexpectedly (should ideally be caught by max_retries logic)
        self.logger.error(f"Exited retry loop unexpectedly for QID {query_id}. This should not happen.")
        return self._handle_final_failure(
            overall_start_time, "Exited retry loop unexpectedly", query_id, dataset_type,
            reasoning_path_taken if reasoning_path_taken != 'unknown_initial' else "path_unknown_at_loop_exit"
        )

    def _timed_process_query(
            self, query: str, context: str, forced_path: Optional[str],
            query_complexity: float, supporting_facts: Optional[List[Tuple[str, int]]],
            query_id: str, dataset_type: Optional[str]
    ) -> Tuple[Any, str]:
        """
        Selects and executes the processing path, respecting active components
        and providing fallbacks if the initially chosen path is unavailable or fails.
        Returns the result and the path taken.
        """
        self.logger.debug(f"SCM._timed_process_query: Processing QID {query_id}, Dataset: '{dataset_type}'")

        # Determine component activity status
        is_symbolic_active = not isinstance(self.hybrid_integrator.symbolic_reasoner, DummySymbolicReasoner)
        is_neural_active = not isinstance(self.hybrid_integrator.neural_retriever, DummyNeuralRetriever)
        self.logger.debug(
            f"SCM._timed_process_query: QID {query_id} - Symbolic Active: {is_symbolic_active}, Neural Active: {is_neural_active}")

        initial_selected_path_by_scm = "hybrid"  # Default
        if not forced_path:
            path_selection_start_time = time.time()
            initial_selected_path_by_scm = self.select_reasoning_path(query_complexity, query_id, dataset_type)
            self.logger.debug(
                f"SCM._timed_process_query: QID {query_id} - Path selection took: {time.time() - path_selection_start_time:.4f}s. SCM Initial Path Choice: {initial_selected_path_by_scm}"
            )

        primary_path_to_try = forced_path if forced_path else initial_selected_path_by_scm
        self.logger.info(
            f"SCM._timed_process_query: QID {query_id} - Primary path to attempt: '{primary_path_to_try}' (Forced: {forced_path is not None})")

        # Define an ordered list of paths to attempt based on the primary path and active components
        paths_to_attempt_execution = []

        if primary_path_to_try == "hybrid":
            if is_symbolic_active and is_neural_active:
                paths_to_attempt_execution.append("hybrid")
            # Fallbacks if hybrid is chosen but not fully active, or if hybrid itself fails
            if is_symbolic_active:  # Symbolic is a common fallback
                paths_to_attempt_execution.append("symbolic")
            if is_neural_active:  # Neural is another common fallback
                paths_to_attempt_execution.append("neural")
        elif primary_path_to_try == "symbolic":
            if is_symbolic_active:
                paths_to_attempt_execution.append("symbolic")
            # Fallbacks if symbolic is chosen (even if not active) or if active symbolic fails
            if is_neural_active:  # If symbolic was chosen (even if dummy), neural is a potential fallback
                paths_to_attempt_execution.append("neural")
            if is_symbolic_active and is_neural_active:  # If symbolic was chosen and active, hybrid is a fallback
                paths_to_attempt_execution.append("hybrid")
        elif primary_path_to_try == "neural":
            if is_neural_active:
                paths_to_attempt_execution.append("neural")
            # Fallbacks
            if is_symbolic_active:
                paths_to_attempt_execution.append("symbolic")
            if is_symbolic_active and is_neural_active:
                paths_to_attempt_execution.append("hybrid")
        else:
            self.logger.error(
                f"SCM._timed_process_query: QID {query_id} - Unknown primary_path_to_try: '{primary_path_to_try}'. Constructing default fallbacks.")
            # Default fallback order if primary path is unknown
            if is_symbolic_active and is_neural_active:
                paths_to_attempt_execution.append("hybrid")
            if is_symbolic_active:
                paths_to_attempt_execution.append("symbolic")
            if is_neural_active:
                paths_to_attempt_execution.append("neural")

        # Remove duplicates while preserving order
        paths_to_attempt_execution = list(dict.fromkeys(paths_to_attempt_execution))

        if not paths_to_attempt_execution:
            # This can happen if, for example, SCM chose "symbolic" but symbolic is disabled, AND neural is also disabled.
            self.logger.error(
                f"SCM._timed_process_query: QID {query_id} - No viable paths to attempt after considering active components. Primary choice: '{primary_path_to_try}'. SymActive: {is_symbolic_active}, NeuActive: {is_neural_active}.")
            error_msg = "No active reasoning components available for the chosen or fallback paths."
            if dataset_type and dataset_type.lower() == 'drop':
                return ({"error": error_msg, "type": "error_object", "spans": [], 'number': '',
                         'date': {'day': '', 'month': '', 'year': ''}, "status": "error",
                         "rationale": error_msg}, "no_active_path")
            else:
                return (f"Error: {error_msg}", "no_active_path")

        self.logger.info(
            f"SCM._timed_process_query: QID {query_id} - Ordered paths to attempt execution: {paths_to_attempt_execution}")

        last_exception = None
        for path_attempt_name in paths_to_attempt_execution:
            self.logger.debug(
                f"SCM._timed_process_query: QID {query_id} - Evaluating attempt for path '{path_attempt_name}' from candidates {paths_to_attempt_execution}"
            )

            # Check if this path is actually viable with current *active* components
            path_is_viable_for_attempt = False
            if path_attempt_name == "hybrid":
                if is_symbolic_active and is_neural_active:
                    path_is_viable_for_attempt = True
            elif path_attempt_name == "symbolic":
                if is_symbolic_active:
                    path_is_viable_for_attempt = True
            elif path_attempt_name == "neural":
                if is_neural_active:
                    path_is_viable_for_attempt = True

            if not path_is_viable_for_attempt:
                self.logger.warning(
                    f"SCM._timed_process_query: QID {query_id} - Path '{path_attempt_name}' is NOT viable with current active components (Sym: {is_symbolic_active}, Neu: {is_neural_active}). Skipping this candidate."
                )
                # If this non-viable path was the SCM's initial choice, it's important to note
                if path_attempt_name == initial_selected_path_by_scm and not forced_path and last_exception is None:
                    last_exception = ValueError(
                        f"Path '{path_attempt_name}' selected by SCM but components are disabled by ablation.")
                continue  # Try next path in the candidate list

            self.logger.info(
                f"SCM._timed_process_query: QID {query_id} - Now attempting execution of viable path: '{path_attempt_name}'")
            try:
                result_obj = self._execute_processing_path(
                    path_attempt_name, query, context, query_complexity,
                    supporting_facts, query_id, dataset_type
                )

                self.logger.info(
                    f"SCM._timed_process_query: QID {query_id} - Path '{path_attempt_name}' execution attempt completed. Result type: {type(result_obj)}"
                )

                # Check if the result from the executed path is an error returned by a dummy component
                # This indicates the path was "executed" but was a dummy.
                if isinstance(result_obj, dict) and \
                        result_obj.get("rationale", "").startswith("[Dummy") and \
                        result_obj.get("status") == "error":
                    dummy_error_msg = result_obj.get('rationale')
                    self.logger.warning(
                        f"SCM._timed_process_query: QID {query_id} - Path '{path_attempt_name}' used a dummy component and returned: {dummy_error_msg}. Will try next path if available.")
                    if last_exception is None:  # Store this as the reason for failure if no prior exception
                        last_exception = ValueError(dummy_error_msg)
                    continue  # Try the next path

                # If it's not a dummy error, this path attempt was successful (even if the result_obj itself indicates a domain error)
                return result_obj, path_attempt_name
            except Exception as e:
                self.logger.warning(
                    f"SCM._timed_process_query: QID {query_id} - Path '{path_attempt_name}' execution raised an EXCEPTION: {str(e)}. Trying next path if available.",
                    exc_info=True
                )
                last_exception = e  # Store the exception
                continue  # Try the next path in paths_to_attempt_execution

        # If loop finishes, all attempted viable paths failed
        self.logger.error(
            f"SCM._timed_process_query: QID {query_id} - All viable candidate paths ({paths_to_attempt_execution}) failed or were skipped.")
        if last_exception:
            # Re-raise the last encountered exception (could be from a dummy or a real execution)
            # This will be handled by the retry logic in process_query_with_fallback
            self.logger.error(
                f"SCM._timed_process_query: QID {query_id} - Re-raising last exception: {str(last_exception)}")
            raise last_exception
        else:
            # This case means no paths in paths_to_attempt_execution were viable or attempted,
            # or they failed silently without exceptions (which shouldn't happen with the dummy error check).
            final_error_msg = "No viable processing path found or all attempted paths failed without a specific exception."
            self.logger.error(f"SCM._timed_process_query: QID {query_id} - {final_error_msg}")
            # Return a structured error for the calling function
            if dataset_type and dataset_type.lower() == 'drop':
                return ({"error": final_error_msg, "type": "error_object", "spans": [], 'number': '',
                         'date': {'day': '', 'month': '', 'year': ''}, "status": "error",
                         "rationale": final_error_msg}, "no_path_executed")
            else:
                return (f"Error: {final_error_msg}", "no_path_executed")

    def select_reasoning_path(
            self,
            query_complexity: float,
            query_id: str,
            dataset_type: Optional[str]
    ) -> str:
        """Selects reasoning path by calling the helper logic or using a fixed path if adaptive logic is disabled."""
        if not self.use_adaptive_logic:
            fixed_path = "hybrid"  # Or your chosen fixed strategy for this ablation
            self.logger.info(
                f"Path Selection for QID {query_id}: Adaptive logic disabled. Using fixed path: {fixed_path}.")
            # Optionally, log factors that would have been used for a more complete picture in path_history
            resource_metrics = self.resource_manager.check_resources()
            decision_factors_for_log = {
                'query_complexity': round(query_complexity, 3),
                'cpu_usage': round(resource_metrics.get('cpu', 0.0) or 0.0, 3),
                'memory_usage': round(resource_metrics.get('memory', 0.0) or 0.0, 3),
                'gpu_usage': round(resource_metrics.get('gpu', 0.0) or 0.0, 3),
                'overall_resource_pressure': round(max(resource_metrics.get('cpu', 0.0) or 0.0,
                                                       resource_metrics.get('memory', 0.0) or 0.0,
                                                       resource_metrics.get('gpu', 0.0) or 0.0), 3),
                'dataset_type': dataset_type or 'unknown',
                'reason': f"Adaptive logic disabled, fixed path '{fixed_path}' selected."
            }
            if hasattr(self, 'path_history'):
                self.path_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query_id': query_id,
                    'chosen_path': fixed_path,
                    'factors': decision_factors_for_log, # Log what the factors were even if not used for decision
                    'reason': decision_factors_for_log['reason']
                })
            return fixed_path

        # Original adaptive logic
        resource_metrics = self.resource_manager.check_resources()
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
            'dataset_type': dataset_type or 'unknown',
            'low_complexity_thr': self.low_complexity_thr,
            'high_complexity_thr': self.high_complexity_thr,
            'low_resource_thr': self.low_resource_thr,
            'high_resource_thr': self.high_resource_thr
        }

        path, reason = _determine_reasoning_path_logic(
            query_complexity,
            self.low_complexity_thr,
            self.high_complexity_thr,
            self.low_resource_thr,
            self.high_resource_thr,
            overall_resource_pressure,
            dataset_type or 'unknown'  # Ensure it's a string
        ) #

        self.logger.info(
            f"Path Selection Decision for QID {query_id}: Chosen Path='{path}' (Reason: {reason}) Factors: {decision_factors}")

        try:
            if hasattr(self, 'path_history'):
                self.path_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query_id': query_id,
                    'chosen_path': path,
                    'factors': decision_factors,
                    'reason': reason,
                }) # [cite: 1454]
        except Exception as hist_err:
            self.logger.error(f"Failed to append to path_history for QID {query_id}: {hist_err}") # [cite: 1455]
        return path

    def _optimize_resources(self):
        """Calls the helper logic for optimizing resources if adaptive logic is enabled."""
        if not self.use_adaptive_logic:
            self.logger.info("Resource Optimization: Adaptive logic disabled. Thresholds remain fixed at their current values.")
            # You might want to ensure they are at *initial defaults* if they could be changed by other means
            # For example:
            # self.low_complexity_thr = 0.4
            # self.high_complexity_thr = 0.8
            # self.low_resource_thr = 0.6
            # self.high_resource_thr = 0.85
            return

        # Original adaptive resource optimization logic
        try:
            resource_metrics = self.resource_manager.check_resources()
            cpu_usage = resource_metrics.get('cpu', 0.0) or 0.0
            memory_usage = resource_metrics.get('memory', 0.0) or 0.0
            gpu_usage = resource_metrics.get('gpu', 0.0) or 0.0
            overall_pressure = max(cpu_usage, memory_usage, gpu_usage)

            current_thresholds = {
                'low_complexity_thr': self.low_complexity_thr,
                'high_complexity_thr': self.high_complexity_thr,
                'low_resource_thr': self.low_resource_thr,
                'high_resource_thr': self.high_resource_thr
            }

            updated_thresholds, adjustments_log_list = _optimize_thresholds_logic(
                current_thresholds,
                overall_pressure,
                dict(self.reasoning_path_stats)
            ) #

            self.low_complexity_thr = updated_thresholds['low_complexity_thr']
            self.high_complexity_thr = updated_thresholds['high_complexity_thr'] # [cite: 1459]
            self.low_resource_thr = updated_thresholds['low_resource_thr']
            self.high_resource_thr = updated_thresholds['high_resource_thr']

            if adjustments_log_list:
                self.logger.info(f"Resource Optimization Adjustments: {', '.join(adjustments_log_list)}")
            else:
                self.logger.info("Resource Optimization: No adjustments needed at this time based on current logic.") # [cite: 1460]
        except Exception as e:
            self.logger.error(f"Error in _optimize_resources: {str(e)}")
            # Revert to default thresholds if optimization fails
            self.low_complexity_thr = 0.4
            self.high_complexity_thr = 0.8
            self.low_resource_thr = 0.6
            self.high_resource_thr = 0.85 # [cite: 1461]
            self.logger.info("Reverted path selection thresholds to default due to optimization error.")

    def _execute_processing_path(
            self, path: str, query: str, context: str, query_complexity: float,
            supporting_facts: Optional[List[Tuple[str, int]]], query_id: str, dataset_type: Optional[str]
    ) -> Any:
        """
        Executes the query processing along the specified path.
        """
        self.logger.debug(f"Executing path '{path}' for QID {query_id}...")
        result_obj: Any = None  # Stores the direct output from the reasoner/integrator

        # Store component timings from HybridIntegrator if it returns them
        sym_time, neu_time, fus_time = 0.0, 0.0, 0.0

        try:
            if path == "symbolic":
                if not hasattr(self.hybrid_integrator,
                               'symbolic_reasoner') or not self.hybrid_integrator.symbolic_reasoner:
                    raise RuntimeError("Symbolic reasoner component is not available.")
                # Assuming symbolic_reasoner.process_query might return a dict with timings
                processed_output = self.hybrid_integrator.symbolic_reasoner.process_query(
                    query, context=context, dataset_type=dataset_type, query_id=query_id
                )
                if isinstance(processed_output,
                              dict) and 'result' in processed_output:  # New: Check for structured output
                    result_obj = processed_output['result']
                    sym_time = processed_output.get('processing_time', 0.0)  # Or a more specific symbolic_time
                else:
                    result_obj = processed_output  # Assume it's the direct answer
                    # sym_time might need to be timed here if not returned by process_query

            elif path == "neural":
                if not hasattr(self.hybrid_integrator,
                               'neural_retriever') or not self.hybrid_integrator.neural_retriever:
                    raise RuntimeError("Neural retriever component is not available.")
                # Assuming neural_retriever.retrieve_answer might return a dict with timings
                processed_output = self.hybrid_integrator.neural_retriever.retrieve_answer(
                    context, query, symbolic_guidance=None, query_complexity=query_complexity, dataset_type=dataset_type
                )
                if isinstance(processed_output,
                              dict) and 'result' in processed_output:  # New: Check for structured output
                    result_obj = processed_output['result']
                    neu_time = processed_output.get('processing_time', 0.0)
                else:
                    result_obj = processed_output
                    # neu_time might need to be timed here

            elif path == "hybrid":
                if not self.hybrid_integrator:
                    raise RuntimeError("Hybrid integrator component is not available.")
                # HybridIntegrator.process_query returns (result, source_str)
                # It also updates self.hybrid_integrator.component_times
                result_content, _ = self.hybrid_integrator.process_query(  # result_content is the actual answer payload
                    query, context, query_complexity=query_complexity, supporting_facts=supporting_facts,
                    query_id=query_id
                )
                result_obj = result_content  # The actual answer
                # Get component times from the HybridIntegrator instance
                sym_time = self.hybrid_integrator.component_times.get('symbolic', 0.0)
                neu_time = self.hybrid_integrator.component_times.get('neural', 0.0)
                fus_time = self.hybrid_integrator.component_times.get('fusion', 0.0)
            else:
                self.logger.error(f"Attempted to execute unknown path '{path}' for QID {query_id}.")
                raise ValueError(f"Unknown processing path: {path}")

            self.logger.debug(f"Path '{path}' execution completed for QID {query_id}. Result type: {type(result_obj)}")

            if result_obj is None:
                self.logger.warning(f"Path '{path}' execution for QID {query_id} returned None.")
                error_msg = f"Error: Path '{path}' returned no result."
                return {"error": error_msg, "type": "error_object", "spans": [], 'number': '',
                        'date': {'day': '', 'month': '', 'year': ''},
                        "status": "error"} if dataset_type == 'drop' else error_msg

            # If result_obj is not already a dict, or if it is but doesn't have timing info,
            # and we have component timings, create/update a dict to include them.
            if not isinstance(result_obj, dict) or not all(
                    k in result_obj for k in ['symbolic_time', 'neural_time', 'fusion_time']):
                if path == "hybrid":  # Only for hybrid are all three times meaningful from integrator
                    if isinstance(result_obj, dict):  # If it's already a dict (like a DROP answer)
                        result_obj['symbolic_time'] = sym_time
                        result_obj['neural_time'] = neu_time
                        result_obj['fusion_time'] = fus_time
                    else:  # If it's a string (like text QA answer)
                        # We need to return a structure that can carry these times,
                        # HybridIntegrator.process_query now returns them so this might be redundant.
                        # For now, we'll assume HybridIntegrator's output `result_obj` (which is `result_content`)
                        # is the primary answer. Timings are logged by process_query_with_fallback.
                        pass  # No change to result_obj if it's a string and not hybrid path.

            return result_obj

        except Exception as e:
            self.logger.exception(f"Critical exception during path execution '{path}' for QID {query_id}: {str(e)}")
            raise e  # Re-raise to be caught by retry logic in process_query_with_fallback

    # --- Methods for Statistics Tracking and Retrieval ---
    # Use the updated versions of these methods provided in the previous step
    def _update_reasoning_path_stats(self, path: str, success: bool, time_taken: Optional[float] = None):
        # (Same as the enhanced version from the previous response)
        if not path or path == 'unknown':
            self.logger.debug(f"Skipping stats update for undefined or 'unknown' path: '{path}'")
            return
        current_stats = self.reasoning_path_stats[path]
        original_count = current_stats.get('count', 0)
        current_stats['count'] += 1
        if success:
            current_stats['success'] += 1
        if time_taken is not None:
            if isinstance(time_taken, (int, float)) and time_taken >= 0:
                current_stats['total_time'] += time_taken
                if current_stats['count'] > 0:
                    current_stats['avg_time'] = current_stats['total_time'] / current_stats['count']
                else:
                    current_stats['avg_time'] = 0.0
            else:
                self.logger.warning(
                    f"Invalid time_taken value ({time_taken}, type: {type(time_taken)}) "
                    f"received for path '{path}'. Ignoring for average time calculation."
                )
        self.logger.debug(
            f"Updated reasoning_path_stats for path '{path}': "
            f"New Count={current_stats['count']} (was {original_count}), "
            f"Successes={current_stats['success']}, "
            f"TotalTime={current_stats['total_time']:.3f}s, "
            f"AvgTime={current_stats['avg_time']:.3f}s"
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        # (Same as the enhanced version from the previous response)
        total_q = self.performance_metrics.get('total_queries', 0)
        successful_queries_count = self.performance_metrics.get('successful_queries', 0)
        total_time_for_successful_queries = 0.0
        for path_name, path_data in self.reasoning_path_stats.items():
            if path_name != 'fallback_error' and path_data.get('success', 0) > 0:
                total_time_for_successful_queries += path_data.get('total_time', 0.0)
        avg_successful_resp_time = 0.0
        if successful_queries_count > 0:
            avg_successful_resp_time = total_time_for_successful_queries / successful_queries_count
        elif total_q > 0:
            self.logger.warning(
                f"Calculating avg_successful_response_time_sec: 'successful_queries' is {successful_queries_count} "
                f"while 'total_queries' is {total_q}. Average time will be 0."
            )
        self.logger.debug(
            f"get_performance_metrics: total_queries={total_q}, successful_queries_count={successful_queries_count}, "
            f"total_time_for_successful_queries_from_reasoning_stats={total_time_for_successful_queries:.3f}, "
            f"calculated_avg_successful_resp_time={avg_successful_resp_time:.3f}"
        )
        if total_time_for_successful_queries == 0 and successful_queries_count > 0:
            self.logger.warning(
                "get_performance_metrics: total_time_for_successful_queries is 0 but successful_queries_count > 0. "
                "This strongly indicates an issue with 'total_time' not being accumulated correctly in 'reasoning_path_stats' "
                "for paths that were part of a successful query resolution, or 'success' flags not being set."
            )
        metrics = {
            'total_queries': total_q,
            'successful_queries': successful_queries_count,
            'success_rate': (successful_queries_count / total_q) * 100 if total_q > 0 else 0.0,
            'error_count': self.performance_metrics.get('error_count', 0),
            'avg_successful_response_time_sec': avg_successful_resp_time,
            'path_chosen_distribution': dict(self.performance_metrics.get('path_usage', defaultdict(int)))
        }
        return metrics

    def get_reasoning_path_stats(self) -> Dict[str, Any]:
        # (Same as the enhanced version from the previous response)
        total_q_processed = self.performance_metrics.get('total_queries', 0)
        denominator_for_percentage = total_q_processed if total_q_processed > 0 else 1
        if not self.reasoning_path_stats:
            self.logger.warning(
                "get_reasoning_path_stats: self.reasoning_path_stats dictionary is empty. "
                "No path execution data has been recorded via _update_reasoning_path_stats."
            )
            return {}
        stats_summary = {}
        found_any_executed_path_data = False
        for path, data in self.reasoning_path_stats.items():
            count = data.get('count', 0)
            success = data.get('success', 0)
            avg_time_sec = data.get('avg_time', 0.0)
            if count > 0:
                found_any_executed_path_data = True
                path_success_rate = (success / count) * 100 if count > 0 else 0.0
                stats_summary[path] = {
                    'execution_count': count, 'success_count': success,
                    'path_success_rate_percent': round(path_success_rate, 2),
                    'avg_time_sec': round(avg_time_sec, 3),
                    'percentage_of_total_queries': round((count / denominator_for_percentage) * 100, 2)
                }
            else:
                self.logger.debug(f"get_reasoning_path_stats: Path '{path}' exists in stats but has 0 execution_count.")
        if not found_any_executed_path_data and total_q_processed > 0:
            self.logger.warning(
                "get_reasoning_path_stats: No paths in self.reasoning_path_stats have an execution_count > 0, "
                f"yet total_queries processed is {total_q_processed}. "
                "This indicates _update_reasoning_path_stats is not being called correctly or not incrementing counts for executed paths."
            )
        elif not found_any_executed_path_data and total_q_processed == 0:
            self.logger.info("get_reasoning_path_stats: No queries processed and no path execution data to summarize.")
        if not stats_summary and total_q_processed > 0:  # If summary is empty but queries were run
            self.logger.warning("get_reasoning_path_stats: Returning an empty summary despite queries being processed. "
                                "Check logs from _update_reasoning_path_stats.")
        return stats_summary

    def _handle_final_failure(self, overall_start_time: float, reason_str: str, query_id: str,
                              dataset_type: Optional[str], last_reasoning_path: str = 'fallback_error') -> Tuple[
        Dict[str, Any], str]:
        """Logs final failure, updates stats, and returns a structured error response tuple."""
        self.logger.error(
            f"Final failure processing QID {query_id} on path '{last_reasoning_path}' after retries or fatal error: {reason_str}")
        processing_time = time.time() - overall_start_time

        # Use last_reasoning_path if specific, otherwise fallback_error
        final_error_path_name = last_reasoning_path if last_reasoning_path not in ['unknown_initial',
                                                                                   'unknown_this_attempt',
                                                                                   'path_unknown_during_error',
                                                                                   'path_unknown_at_final_failure',
                                                                                   'path_unknown_at_loop_exit'] else 'fallback_error'

        error_result_payload: Union[str, Dict]
        if dataset_type == 'drop':
            error_result_payload = {"error": reason_str, "type": "error_object", "spans": [], 'number': '',
                                    'date': {'day': '', 'month': '', 'year': ''}, "status": "failed"}
        else:
            error_result_payload = f"Error: {reason_str}"

        # Update stats for this final failure path
        self._update_reasoning_path_stats(final_error_path_name, success=False, time_taken=processing_time)

        formatted_error_response = self.aggregator.format_response({
            'query_id': query_id,
            'result': error_result_payload,  # This is the core error content
            'error': reason_str,  # Explicit error message
            'status': 'failed',
            'reasoning_path': final_error_path_name,
            'processing_time': processing_time,
            'retries': self.error_retry_limit,  # Max retries attempted
            'dataset_type': dataset_type
        })
        return formatted_error_response, final_error_path_name