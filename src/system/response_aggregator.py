# src/system/response_aggregator.py

import logging
from typing import Any, Dict, Optional, Union  # Added Union


# logger = logging.getLogger(__name__) # This is fine if __name__ is 'src.system.response_aggregator'
# Or define explicitly as below for clarity

class UnifiedResponseAggregator:
    """
    Aggregates responses with optional explanations, supporting both HotpotQA (string) and DROP (dict) results.
    Ensures consistent schema and type handling in the final formatted response.
    """

    def __init__(self, include_explanations: bool = False):
        self.include_explanations = include_explanations
        self.logger = logging.getLogger("UnifiedResponseAggregator")  # Explicit logger name
        self.logger.setLevel(logging.INFO)  # Set a default level, can be overridden by main config

    def aggregate(self, result: Any, source: str, confidence: float = 1.0,
                  debug_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggregates a single result with its source and optional debug information.
        This method seems more like a helper to package intermediate results if needed,
        while format_response handles the final structure from SystemControlManager's data.
        """
        output = {
            'result': result,
            'source': source,  # This is the reasoning path (symbolic, neural, hybrid)
            'confidence': confidence,
            'reasoning_path': source  # Redundant with 'source', but kept from original
        }
        if self.include_explanations and debug_info:
            output['debug_info'] = debug_info
            # Prefer more specific fusion strategy keys if available
            if 'fusion_strategy_text' in debug_info:
                output['fusion_strategy'] = debug_info['fusion_strategy_text']
            elif 'fusion_strategy_drop' in debug_info:
                output['fusion_strategy'] = debug_info['fusion_strategy_drop']
            # Pass through component timings if available in debug_info
            output['symbolic_time'] = debug_info.get('symbolic_time', 0.0)
            output['neural_time'] = debug_info.get('neural_time', 0.0)
            output['fusion_time'] = debug_info.get('fusion_time', 0.0)
        return output

    def format_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats the comprehensive data bundle from SystemControlManager into a final
        response dictionary suitable for the user or further evaluation.

        'data' dictionary is expected to contain:
            'query_id',
            'result' (the actual answer object/string from integrator/reasoner),
            'processing_time',
            'resource_usage',
            'reasoning_path',
            'retries',
            'status' (from SystemControlManager: 'success' or 'failed'),
            'dataset_type',
            'error' (optional, string message of the final error if status is 'failed'),
            'symbolic_time' (optional),
            'neural_time' (optional),
            'fusion_time' (optional)
        """
        query_id = data.get('query_id', 'unknown_qid')
        self.logger.debug(f"Formatting response for QID {query_id}. Input data status: {data.get('status')}")

        # Initialize the base structure of the formatted response
        formatted_response = {
            'query_id': query_id,
            'result': None,  # To be populated based on type and content
            'processing_time': data.get('processing_time', 0.0),
            'resource_usage': data.get('resource_usage', {}),
            'reasoning_path': data.get('reasoning_path', 'unknown_path_in_aggregator'),
            'retries': data.get('retries', 0),
            'status': data.get('status', 'error'),  # Default to error if not specified
            'error': data.get('error'),  # Preserve error message from SCM
            'symbolic_time': data.get('symbolic_time', 0.0),
            'neural_time': data.get('neural_time', 0.0),
            'fusion_time': data.get('fusion_time', 0.0)
        }

        raw_result_payload = data.get('result')
        dataset_type_local = data.get('dataset_type')
        # The 'status' from SCM is the overall status of the query processing attempt
        current_overall_status = formatted_response['status']

        if current_overall_status == 'failed':
            # If the entire process failed, ensure the error message is primary.
            # The raw_result_payload might be the specific error object or string.
            self.logger.warning(
                f"QID {query_id}: Processing was marked as 'failed' by SystemControlManager. Error: {formatted_response['error']}")
            if dataset_type_local == 'drop':
                # Ensure even a failed DROP response has the basic structure
                error_rationale = formatted_response['error'] or "Processing failed"
                if isinstance(raw_result_payload, dict) and raw_result_payload.get(
                        'error'):  # If payload is an error dict
                    error_rationale = raw_result_payload.get('rationale', error_rationale)

                formatted_response['result'] = {
                    'number': '', 'spans': [], 'date': {'day': '', 'month': '', 'year': ''},
                    'status': 'error', 'confidence': 0.0,
                    'rationale': error_rationale, 'type': 'error_system_level'
                }
            else:  # Text QA
                formatted_response['result'] = formatted_response['error'] or "Error: Processing failed."

        elif raw_result_payload is None:
            self.logger.warning(f"QID {query_id}: 'result' payload is None. Setting to error/empty.")
            formatted_response['status'] = 'failed'  # Mark as failed if no result payload
            formatted_response['error'] = formatted_response.get('error', "Error: No result payload generated.")
            if dataset_type_local == 'drop':
                formatted_response['result'] = {
                    'number': '', 'spans': [], 'date': {'day': '', 'month': '', 'year': ''},
                    'status': 'error', 'confidence': 0.0,
                    'rationale': formatted_response['error'], 'type': 'error_no_payload'
                }
            else:
                formatted_response['result'] = formatted_response['error']

        # If processing was successful or payload exists, format it
        else:
            if dataset_type_local == 'drop':
                empty_drop_shell = {'number': '', 'spans': [], 'date': {'day': '', 'month': '', 'year': ''}}

                if isinstance(raw_result_payload, dict):
                    # Start with defaults and update from payload
                    final_drop_result = {
                        **empty_drop_shell,  # ensure all keys exist
                        'status': raw_result_payload.get('status', 'success'),  # Prefer status from payload
                        'confidence': raw_result_payload.get('confidence', 0.5),
                        'rationale': raw_result_payload.get('rationale', 'No rationale provided.'),
                        'type': raw_result_payload.get('type', 'unknown')  # Operational type
                    }
                    # Copy specific answer fields
                    if raw_result_payload.get('number') is not None:  # Could be 0 or native type
                        final_drop_result['number'] = raw_result_payload.get('number')
                    if raw_result_payload.get('spans') is not None:  # Could be empty list
                        final_drop_result['spans'] = raw_result_payload.get('spans')
                    if isinstance(raw_result_payload.get('date'), dict):
                        final_drop_result['date'] = {
                            'day': str(raw_result_payload['date'].get('day', '')),
                            'month': str(raw_result_payload['date'].get('month', '')),
                            'year': str(raw_result_payload['date'].get('year', ''))
                        }

                    # If the payload itself indicates an error, reflect it
                    if final_drop_result['status'] == 'error' or raw_result_payload.get('error'):
                        formatted_response['status'] = 'failed'  # Mark overall as failed
                        if formatted_response['error'] is None:
                            formatted_response['error'] = final_drop_result.get('rationale',
                                                                                'Error in DROP result payload.')

                    # Type enforcement for 'number' if it's a string representation from an earlier stage
                    # (Ideally, HybridIntegrator and Reasoners already provide native types)
                    current_number_val = final_drop_result.get('number')
                    if isinstance(current_number_val, str) and current_number_val.strip():
                        try:
                            num_val_float = float(current_number_val)
                            final_drop_result['number'] = int(
                                num_val_float) if num_val_float.is_integer() else num_val_float
                        except ValueError:
                            self.logger.warning(
                                f"QID {query_id}: Aggregator found string number '{current_number_val}' that is not parsable. Keeping string or setting empty.")
                            # If it's not numeric, it shouldn't be in 'number'. HybridIntegrator should ensure this.
                            # For safety, if it's a non-numeric string, make it empty string.
                            if not current_number_val.replace('.', '', 1).replace('-', '', 1).isdigit():
                                final_drop_result['number'] = ""

                    formatted_response['result'] = final_drop_result

                elif isinstance(raw_result_payload, str) and raw_result_payload.lower().startswith("error:"):
                    self.logger.warning(
                        f"QID {query_id}: Received error string for DROP result: '{raw_result_payload}'. Formatting as error dict.")
                    formatted_response['result'] = {**empty_drop_shell, 'status': 'error', 'confidence': 0.0,
                                                    'rationale': raw_result_payload, 'type': 'error_string_payload'}
                    formatted_response['status'] = 'failed'
                    if formatted_response['error'] is None: formatted_response['error'] = raw_result_payload
                else:
                    self.logger.error(
                        f"QID {query_id}: Unexpected result payload format for DROP: {type(raw_result_payload)}. Content: '{str(raw_result_payload)[:200]}'. Formatting as default error.")
                    formatted_response['result'] = {**empty_drop_shell, 'status': 'error', 'confidence': 0.0,
                                                    'rationale': f"Unexpected result structure: {str(raw_result_payload)[:100]}",
                                                    'type': 'error_structure'}
                    formatted_response['status'] = 'failed'
                    if formatted_response['error'] is None: formatted_response[
                        'error'] = f"Unexpected result structure for DROP: {str(raw_result_payload)[:100]}"

            else:  # For non-DROP datasets (e.g., HotpotQA, text-based QA)
                if not isinstance(raw_result_payload, str):
                    self.logger.warning(
                        f"QID {query_id}: Text QA result is not a string ({type(raw_result_payload)}). Converting. Payload: '{str(raw_result_payload)[:100]}'")
                    formatted_response['result'] = str(raw_result_payload)
                    if formatted_response['status'] == 'success':  # If SCM said success, but type was odd
                        formatted_response['status'] = 'success_with_conversion'
                else:
                    formatted_response['result'] = raw_result_payload

                if formatted_response['result'].lower().startswith("error:"):
                    formatted_response['status'] = 'failed'
                    if formatted_response['error'] is None: formatted_response['error'] = formatted_response['result']

        if self.include_explanations:
            formatted_response["explanation"] = self._generate_detailed_explanation(formatted_response)

        # Final status consistency check
        if formatted_response['status'] not in ['success', 'failed', 'success_with_conversion']:
            if formatted_response.get('error'):
                formatted_response['status'] = 'failed'
            elif formatted_response.get('result') is not None:  # Check if result exists
                # For DROP, check if the result content is non-empty and not an error status within result
                is_drop_result_empty_or_error = False
                if dataset_type_local == 'drop' and isinstance(formatted_response['result'], dict):
                    res_dict = formatted_response['result']
                    is_drop_result_empty_or_error = (
                            res_dict.get('status') == 'error' or
                            (not res_dict.get('number') and not res_dict.get('spans') and not any(
                                res_dict.get('date', {}).values()))
                    )
                if not is_drop_result_empty_or_error:
                    formatted_response['status'] = 'success'
                else:
                    formatted_response['status'] = 'failed'
                    if formatted_response['error'] is None and is_drop_result_empty_or_error:
                        formatted_response['error'] = "Result indicates error or is empty."
            else:
                formatted_response['status'] = 'failed'  # Default to failed if no result and no explicit error
                if formatted_response['error'] is None: formatted_response[
                    'error'] = "Processing completed with no result."

        self.logger.debug(
            f"QID {query_id}: Final formatted response status after all checks: {formatted_response['status']}")
        return formatted_response

    def _generate_detailed_explanation(self, data: Dict[str, Any]) -> str:
        """
        Generates a detailed explanation string from the provided data dictionary.
        Ensures graceful handling of missing keys and correct formatting.
        """
        parts = []
        path = data.get('reasoning_path', 'unknown_path')
        parts.append(f"Reasoning Approach: {path}")

        proc_time = data.get('processing_time')
        if isinstance(proc_time, (int, float)):
            parts.append(f"Processing Time: {proc_time:.3f}s")

        resource_usage = data.get('resource_usage')
        if isinstance(resource_usage, dict):
            usage_parts = []
            for resource, usage_val in resource_usage.items():
                if isinstance(usage_val, (int, float)):
                    # Show sign for delta, and format as percentage if values are 0-1 representing delta percentage
                    # Assuming values from ResourceManager.get_resource_delta are already fractions.
                    usage_str = f"{usage_val * 100:+.1f}%" if -1 <= usage_val <= 1 else f"{usage_val:.2f}"
                    usage_parts.append(f"{resource.capitalize()}: {usage_str}")
                else:
                    usage_parts.append(f"{resource.capitalize()}: {usage_val}")
            if usage_parts:
                parts.append(f"Resource Delta: [{', '.join(usage_parts)}]")

        # Component timings from the input data dictionary
        timings = []
        for key, label in [('symbolic_time', 'Sym'), ('neural_time', 'Neu'), ('fusion_time', 'Fus')]:
            time_val = data.get(key)  # Use .get() for safety
            if isinstance(time_val, (int, float)) and time_val > 0.0001:  # Only show if significant
                timings.append(f"{label}: {time_val:.3f}s")
        if timings:
            parts.append(f"Component Times: [{', '.join(timings)}]")

        retries = data.get('retries', 0)
        if retries > 0:
            parts.append(f"Retries Attempted: {retries}")

        status = data.get('status', 'unknown')
        if status == 'failed':
            err_msg = data.get('error', 'Unknown reason')
            # Ensure err_msg is a string and truncate if too long
            err_msg_str = str(err_msg)[:150] + "..." if len(str(err_msg)) > 150 else str(err_msg)
            parts.append(f"Status: Failed ({err_msg_str})")
        elif status == 'success_with_conversion':
            parts.append("Status: Success (output type converted)")

        return " | ".join(parts) if parts else "No detailed explanation available."