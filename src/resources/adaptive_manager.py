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
        """Record performance metrics for a query type."""
        self.performance_metrics[query_type].append(metrics)
        if len(self.performance_metrics[query_type]) > self.window_size:
            self.performance_metrics[query_type].pop(0)

        # Trigger learning if we have enough data
        if len(self.performance_metrics[query_type]) >= self.window_size // 2:
            self._learn_thresholds(query_type)

    def _learn_thresholds(self, query_type):
        """Learn optimal thresholds based on performance history."""
        metrics = self.performance_metrics[query_type]

        # Calculate average resource usage for successful queries
        successful_metrics = [m for m in metrics if m.get('success', False)]
        if not successful_metrics:
            return

        avg_cpu = np.mean([m['cpu_usage'] for m in successful_metrics])
        avg_gpu = np.mean([m['gpu_usage'] for m in successful_metrics])
        avg_memory = np.mean([m['memory_usage'] for m in successful_metrics])

        # Update thresholds with safety margin
        self.pattern_thresholds[query_type] = {
            'cpu': min(0.95, avg_cpu * 1.2),
            'gpu': min(0.95, avg_gpu * 1.2),
            'memory': min(0.90, avg_memory * 1.2)
        }

    def get_thresholds(self, query_type):
        """Get learned thresholds for a query type."""
        return self.pattern_thresholds[query_type]
