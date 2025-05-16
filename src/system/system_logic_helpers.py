# src/system/system_logic_helpers.py

import logging
from typing import Any, Dict, Tuple, List  # Added List
from collections import defaultdict  # Added defaultdict
import numpy as np  # Added numpy for _optimize_resources if it uses it directly


# It's good practice for helper modules to use their own logger or be passed one
# helper_logger = logging.getLogger("SystemLogicHelpers")

def _determine_reasoning_path_logic(
        query_complexity: float,
        low_complexity_thr: float,
        high_complexity_thr: float,
        low_resource_thr: float,
        high_resource_thr: float,
        overall_resource_pressure: float,  # Added this explicit parameter
        dataset_type: str,
        # query_id: str, # query_id is mainly for logging within the SCM method
        # logger_instance: logging.Logger # Pass logger if direct logging is needed here
) -> Tuple[str, str]:  # Returns path and reason
    """Contains the core logic for selecting a reasoning path."""
    path = "hybrid"  # Default
    reason = "Default choice: suitable for balanced complexity/resources."

    dt_lower = dataset_type.lower() if dataset_type else 'unknown'

    if dt_lower == 'drop':
        reason = "Default for DROP: Balanced complexity/resources favor hybrid."
        if query_complexity < low_complexity_thr and overall_resource_pressure < low_resource_thr:
            path = "symbolic"
            reason = f"DROP dataset: Low complexity (<{low_complexity_thr:.2f}) and low resources (<{low_resource_thr:.2f}) favor symbolic."
        elif query_complexity > high_complexity_thr or overall_resource_pressure > high_resource_thr:
            path = "neural"
            if query_complexity > high_complexity_thr:
                reason = f"DROP dataset: High query complexity (> {high_complexity_thr:.2f}) favors neural."
            else:
                reason = f"DROP dataset: High resource pressure (> {high_resource_thr:.2f}) favors neural."
        else:  # Conditions for hybrid for DROP (explicitly stated)
            reason = f"DROP dataset: Balanced complexity ({query_complexity:.3f}) and resources ({overall_resource_pressure:.3f}) favor hybrid."

    else:  # Logic for text-based QA (e.g., HotpotQA)
        reason = "Default for Text QA: Balanced complexity/resources favor hybrid."
        if query_complexity < low_complexity_thr and overall_resource_pressure < low_resource_thr:
            path = "symbolic"
            reason = f"Text QA: Low complexity (<{low_complexity_thr:.2f}) and low resources (<{low_resource_thr:.2f}) favor symbolic."
        elif query_complexity >= high_complexity_thr or overall_resource_pressure >= high_resource_thr:
            path = "neural"
            if query_complexity >= high_complexity_thr:
                reason = f"Text QA: High query complexity (>= {high_complexity_thr:.2f}) favors neural."
            else:
                reason = f"Text QA: High resource pressure (>= {high_resource_thr:.2f}) favors neural."
        else:  # Conditions for hybrid for Text QA (explicitly stated)
            reason = f"Text QA: Balanced complexity ({query_complexity:.3f}) and resources ({overall_resource_pressure:.3f}) favor hybrid."

    return path, reason


def _optimize_thresholds_logic(
        current_thresholds: Dict[str, float],
        overall_pressure: float,  # Pass the pre-calculated overall pressure
        reasoning_path_stats: Dict[str, Dict[str, Any]],  # Pass the stats dict
        # logger_instance: logging.Logger # Pass logger if direct logging is needed here
) -> Tuple[Dict[str, float], List[str]]:  # Returns updated_thresholds, adjustments_log
    """Contains the core logic for optimizing path selection thresholds."""

    # Make a copy to modify
    updated_thresholds = current_thresholds.copy()
    adjustments_log = []  # Log of changes made

    path_performance = {}
    for path, stats_data in reasoning_path_stats.items():  # Changed 'stats' to 'stats_data'
        count = stats_data.get('count', 0)
        if count > 0:
            success_rate = (stats_data.get('success', 0) / count) * 100
            avg_time = stats_data.get('avg_time', 0.0)
            path_performance[path] = {'success_rate': success_rate, 'avg_time': avg_time}
        else:
            path_performance[path] = {'success_rate': 0.0, 'avg_time': 0.0}

    # Adjust thresholds based on resource usage
    if overall_pressure > 0.9:
        if updated_thresholds['low_complexity_thr'] < 0.6:
            updated_thresholds['low_complexity_thr'] += 0.05
            adjustments_log.append(
                f"Increased low_complexity_thr to {updated_thresholds['low_complexity_thr']:.2f} (high resource pressure)")
        if updated_thresholds['high_complexity_thr'] > 0.6:
            updated_thresholds['high_complexity_thr'] -= 0.05
            adjustments_log.append(
                f"Decreased high_complexity_thr to {updated_thresholds['high_complexity_thr']:.2f} (high resource pressure)")
    elif overall_pressure < 0.3:
        if updated_thresholds['low_complexity_thr'] > 0.2:
            updated_thresholds['low_complexity_thr'] -= 0.05
            adjustments_log.append(
                f"Decreased low_complexity_thr to {updated_thresholds['low_complexity_thr']:.2f} (low resource pressure)")
        if updated_thresholds['high_complexity_thr'] < 0.9:
            updated_thresholds['high_complexity_thr'] += 0.05
            adjustments_log.append(
                f"Increased high_complexity_thr to {updated_thresholds['high_complexity_thr']:.2f} (low resource pressure)")

    # Adjust based on path performance
    if path_performance.get('neural', {}).get('success_rate', 100) < 50 and \
            path_performance.get('neural', {}).get('avg_time', 0) > 1.0:
        if updated_thresholds['high_complexity_thr'] > 0.6:
            updated_thresholds['high_complexity_thr'] -= 0.05
            adjustments_log.append(
                f"Decreased high_complexity_thr to {updated_thresholds['high_complexity_thr']:.2f} (poor neural perf)")

    if path_performance.get('symbolic', {}).get('success_rate', 100) < 50 and \
            path_performance.get('symbolic', {}).get('avg_time', 0) > 1.0:
        if updated_thresholds['low_complexity_thr'] > 0.2:
            updated_thresholds['low_complexity_thr'] -= 0.05
            adjustments_log.append(
                f"Decreased low_complexity_thr to {updated_thresholds['low_complexity_thr']:.2f} (poor symbolic perf)")

    return updated_thresholds, adjustments_log