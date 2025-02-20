# src/resources/adaptive_manager.py

import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import logging


class AdaptiveManager:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.query_patterns = defaultdict(list)
        self.resource_history = []
        self.performance_metrics = defaultdict(list)
        self.logger = logging.getLogger("AdaptiveManager")

        # Thresholds for different query types
        self.pattern_thresholds = defaultdict(
            lambda: {
                'cpu': 0.85,
                'gpu': 0.95,
                'memory': 0.85
            }
        )

    def update_metrics(self, query_type, metrics):
        """
        Record performance metrics for a query type.
        """
        self.performance_metrics[query_type].append(metrics)

    def analyze_patterns(self):
        """
        Analyze stored query patterns and performance metrics to suggest adjustments.
        """
        analysis = {}
        for q_type, metrics_list in self.performance_metrics.items():
            if metrics_list:
                avg_latency = np.mean([m.get('latency', 0) for m in metrics_list])
                avg_accuracy = np.mean([m.get('accuracy', 0) for m in metrics_list])
                analysis[q_type] = {
                    'avg_latency': avg_latency,
                    'avg_accuracy': avg_accuracy,
                    'trend_latency': self._calculate_trend([m.get('latency', 0) for m in metrics_list]),
                    'trend_accuracy': self._calculate_trend([m.get('accuracy', 0) for m in metrics_list]),
                }
        return analysis

    def recommend_adjustments(self):
        """
        Generate recommendations based on analysis. E.g., if latency is too high,
        we might reduce concurrency or reallocate resources.
        """
        analysis = self.analyze_patterns()
        recommendations = []

        for q_type, stats in analysis.items():
            # Example logic: if avg_latency is above 2.0 seconds, reduce concurrency
            if stats['avg_latency'] > 2.0:
                recommendations.append({
                    'type': q_type,
                    'action': 'reduce_concurrency',
                    'reason': 'High average latency'
                })

            # If accuracy is below 0.7, consider adjusting retrieval parameters
            if stats['avg_accuracy'] < 0.7:
                recommendations.append({
                    'type': q_type,
                    'action': 'adjust_retrieval',
                    'reason': 'Low average accuracy'
                })

        return recommendations

    def update_resource_history(self, usage):
        """
        Keep a record of system resource usage.
        """
        self.resource_history.append(usage)
        if len(self.resource_history) > self.window_size:
            self.resource_history.pop(0)

    def analyze_resource_usage(self):
        """
        Analyze resource usage trends (CPU, GPU, memory).
        """
        if not self.resource_history:
            return {}

        usage_array = np.array([
            [entry['cpu'], entry['gpu'], entry['memory']]
            for entry in self.resource_history
        ])
        mean_usage = np.mean(usage_array, axis=0)
        trend_cpu = self._calculate_trend(usage_array[:, 0])
        trend_gpu = self._calculate_trend(usage_array[:, 1])
        trend_mem = self._calculate_trend(usage_array[:, 2])

        return {
            'mean_cpu': mean_usage[0],
            'mean_gpu': mean_usage[1],
            'mean_memory': mean_usage[2],
            'trend_cpu': trend_cpu,
            'trend_gpu': trend_gpu,
            'trend_memory': trend_mem
        }

    def _calculate_trend(self, values):
        """
        Calculate a simple slope to measure the trend of the data.
        """
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        try:
            slope = np.polyfit(x, y, 1)[0]
        except np.linalg.LinAlgError:
            slope = 0.0
        return slope
