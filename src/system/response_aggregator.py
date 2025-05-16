# src/system/response_aggregator.py

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__) # Use its own logger if needed, or pass one

class UnifiedResponseAggregator:
    """
    Aggregates responses with optional explanations, supporting both HotpotQA (string) and DROP (dict) results.
    """
    def __init__(self, include_explanations: bool = False):
        self.include_explanations = include_explanations
        # It's good practice for classes to have their own logger instance.
        # If it needs to share the logger from SystemControlManager, it can be passed in.
        self.logger = logging.getLogger("UnifiedResponseAggregator")
        # Or, self.logger = logger # if logger is passed in __init__

    def aggregate(self, result: Any, source: str, confidence: float = 1.0, debug_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        output = {
            'result': result,
            'source': source,
            'confidence': confidence,
            'reasoning_path': source
        }
        if self.include_explanations and debug_info:
            output['debug_info'] = debug_info
            if 'fusion_strategy_text' in debug_info:
                output['fusion_strategy'] = debug_info['fusion_strategy_text']
            elif 'fusion_strategy_drop' in debug_info:
                output['fusion_strategy'] = debug_info['fusion_strategy_drop']
            output['symbolic_time'] = debug_info.get('symbolic_time', 0.0)
            output['neural_time'] = debug_info.get('neural_time', 0.0)
            output['fusion_time'] = debug_info.get('fusion_time', 0.0)
        return output

    def format_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'result' not in data:
            self.logger.warning(f"Formatting response for QID {data.get('query_id', 'unknown')}: 'result' key missing. Setting default error.")
            data['result'] = data.get('error', "Error: No result generated.")

        result = data['result']
        dataset_type_local = data.get('dataset_type') # Local variable for clarity

        if dataset_type_local == 'drop':
            if isinstance(result, dict):
                result.setdefault('number', '')
                result.setdefault('spans', [])
                result.setdefault('date', {'day': '', 'month': '', 'year': ''})
                if 'error' in result:
                    result.setdefault('type', 'error_object')
            elif isinstance(result, str) and result.startswith("Error:"):
                error_msg = result
                data['result'] = {'error': error_msg, 'type': 'error_object', 'spans': [], 'number': '', 'date': {'day':'','month':'','year':''}}
                self.logger.warning(f"Converted error string to DROP dict for QID {data.get('query_id', 'unknown')}")
            else:
                self.logger.warning(f"Unexpected result format for DROP QID {data.get('query_id', 'unknown')}: {result}. Wrapping as error.")
                error_msg = f"Unexpected result structure: {str(result)[:100]}"
                data['result'] = {'error': error_msg, 'type': 'error_object', 'spans': [], 'number': '', 'date': {'day':'','month':'','year':''}}
        elif not isinstance(result, str): # For non-DROP or if dataset_type is not 'drop'
            self.logger.warning(f"Unexpected result type for Text QA QID {data.get('query_id', 'unknown')}: {type(result)}. Converting to string.")
            data['result'] = str(result)

        if self.include_explanations:
            data.setdefault("explanation", self._generate_detailed_explanation(data))

        data.setdefault('query_id', 'unknown')
        data.setdefault('processing_time', 0.0)
        data.setdefault('resource_usage', {})
        data.setdefault('reasoning_path', 'unknown')
        data.setdefault('retries', 0)
        data.setdefault('status', 'unknown')
        return data

    def _generate_detailed_explanation(self, data: Dict[str, Any]) -> str:
        parts = []
        path = data.get('reasoning_path', 'unknown')
        parts.append(f"Reasoning Approach: {path}")

        proc_time = data.get('processing_time')
        if isinstance(proc_time, (int, float)):
             parts.append(f"Processing Time: {proc_time:.3f}s")

        resource_usage = data.get('resource_usage')
        if isinstance(resource_usage, dict):
            usage_parts = []
            for resource, usage_val in resource_usage.items(): # Renamed 'usage' to 'usage_val'
                if isinstance(usage_val, (int, float)):
                    usage_parts.append(f"{resource.capitalize()}: {usage_val:.2f} delta")
                else:
                    usage_parts.append(f"{resource.capitalize()}: {usage_val}")
            if usage_parts:
                 parts.append(f"Resource Delta: [{', '.join(usage_parts)}]")

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