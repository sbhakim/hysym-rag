# src/utils/metrics_collector.py

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging
import json
import pandas as pd
from pathlib import Path
import torch


class MetricsCollector:
    """
    Comprehensive metrics collection and analysis system for HySym-RAG.
    Designed specifically for academic evaluation and research paper results.
    """

    def __init__(self,
                 metrics_dir: str = "metrics",
                 experiment_name: Optional[str] = None,
                 save_frequency: int = 100):
        """
        Initialize metrics collector with academic focus.

        Args:
            metrics_dir: Directory for storing metrics data
            experiment_name: Name of current experiment run
            save_frequency: Frequency of metrics persistence
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_frequency = save_frequency

        # Initialize logging
        self.logger = logging.getLogger("MetricsCollector")
        self.logger.setLevel(logging.INFO)

        # Initialize comprehensive metrics storage
        self.metrics = {
            # Query-specific metrics (stored as a dictionary keyed by query count)
            'query_metrics': defaultdict(dict),

            # Reasoning path analysis
            'reasoning_paths': defaultdict(list),

            # Resource utilization tracking
            'resource_usage': defaultdict(list),

            # Performance metrics
            'performance': defaultdict(list),

            # Timing analysis
            'timing': defaultdict(list),

            # Error tracking
            'errors': defaultdict(list)
        }

        # Statistical aggregates
        self.statistical_metrics = {
            'confidence_intervals': {},
            'correlations': {},
            'distributions': {}
        }

        # Initialize counters
        self.query_count = 0
        self.start_time = datetime.now()

    def collect_query_metrics(self,
                              query: str,
                              prediction: str,
                              ground_truth: Optional[str],
                              reasoning_path: str,
                              processing_time: float,
                              resource_usage: Dict[str, float],
                              complexity_score: float) -> None:
        """
        Collect comprehensive metrics for a single query execution.

        Args:
            query: Input query
            prediction: System's prediction
            ground_truth: Optional ground truth
            reasoning_path: Path taken (symbolic/hybrid/neural)
            processing_time: Query processing time
            resource_usage: Resource utilization metrics
            complexity_score: Query complexity score
        """
        timestamp = datetime.now()

        # Collect basic metrics
        query_metrics = {
            'timestamp': timestamp.isoformat(),
            'query_length': len(query),
            'prediction_length': len(prediction),
            'processing_time': processing_time,
            'complexity_score': complexity_score,
            'reasoning_path': reasoning_path
        }

        # Resource usage metrics
        query_metrics.update({
            f'resource_{k}': v for k, v in resource_usage.items()
        })

        # Performance metrics if ground truth available
        if ground_truth:
            query_metrics.update(self._calculate_performance_metrics(
                prediction, ground_truth
            ))

        # Update metrics collections
        self.metrics['query_metrics'][self.query_count] = query_metrics
        self.metrics['reasoning_paths'][reasoning_path].append(query_metrics)

        # Update resource tracking
        for resource, value in resource_usage.items():
            self.metrics['resource_usage'][resource].append(value)

        # Update timing metrics
        self.metrics['timing']['processing_times'].append(processing_time)

        # Increment query counter
        self.query_count += 1

        # Save metrics if needed
        if self.query_count % self.save_frequency == 0:
            self._save_metrics()

    def generate_academic_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report for academic paper.
        Includes statistical analysis and confidence intervals.
        """
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics()

        # Analyze reasoning paths
        reasoning_analysis = self._analyze_reasoning_paths()

        # Calculate resource efficiency
        efficiency_analysis = self._analyze_resource_efficiency()

        # Generate statistical significance tests
        statistical_tests = self._perform_statistical_tests()

        return {
            'experiment_summary': {
                'total_queries': self.query_count,
                'duration': (datetime.now() - self.start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            },
            'performance_metrics': overall_stats,
            'reasoning_analysis': reasoning_analysis,
            'efficiency_metrics': efficiency_analysis,
            'statistical_analysis': statistical_tests,
            'confidence_intervals': self.statistical_metrics['confidence_intervals']
        }

    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for all collected metrics.
        """
        stats = {}

        # Processing time statistics
        processing_times = self.metrics['timing']['processing_times']
        stats['processing_time'] = {
            'mean': float(np.mean(processing_times)),
            'std': float(np.std(processing_times)),
            'median': float(np.median(processing_times)),
            'percentile_95': float(np.percentile(processing_times, 95))
        }

        # Resource usage statistics
        for resource, values in self.metrics['resource_usage'].items():
            stats[f'resource_{resource}'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'peak': float(np.max(values))
            }

        # Query complexity statistics
        complexities = [m['complexity_score'] for m in self.metrics['query_metrics'].values()]
        hist, bin_edges = np.histogram(complexities, bins=10)
        stats['query_complexity'] = {
            'mean': float(np.mean(complexities)),
            'std': float(np.std(complexities)),
            'distribution': {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        }

        return stats

    def _calculate_success_rate(self, path_metrics: List[Dict[str, Any]]) -> float:
        """
        Calculate success rate for a given set of path metrics.
        """
        if not path_metrics:
            return 0.0
        successes = sum(1 for m in path_metrics if
                        m.get('performance_metrics', {}).get('success', False))
        return (successes / len(path_metrics)) * 100 if path_metrics else 0.0

    def _analyze_reasoning_paths(self) -> Dict[str, Any]:
        """
        Analyze effectiveness of different reasoning paths.
        """
        path_analysis = {}

        for path, metrics in self.metrics['reasoning_paths'].items():
            path_analysis[path] = {
                'usage_count': len(metrics),
                'usage_percentage': (len(metrics) / self.query_count) * 100,
                'avg_processing_time': float(np.mean([m['processing_time'] for m in metrics])),
                'avg_complexity': float(np.mean([m['complexity_score'] for m in metrics])),
                'success_rate': self._calculate_success_rate(metrics)
            }

            # Add resource efficiency for each path
            path_analysis[path]['resource_efficiency'] = self._calculate_path_efficiency(metrics)

        return path_analysis

    def _analyze_resource_efficiency(self) -> Dict[str, Any]:
        """
        Analyze resource utilization and efficiency metrics.
        """
        efficiency_metrics = {}

        # Calculate overall resource efficiency
        for resource, values in self.metrics['resource_usage'].items():
            efficiency_metrics[resource] = {
                'mean_usage': float(np.mean(values)),
                'peak_usage': float(np.max(values)),
                'efficiency_score': float(1 - (np.mean(values) / np.max(values))) if np.max(values) != 0 else None
            }

        # Calculate efficiency trends
        efficiency_metrics['trends'] = self._calculate_efficiency_trends()

        return efficiency_metrics

    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical significance tests on collected metrics.
        """
        tests = {}

        # Perform t-tests between reasoning paths (placeholder implementation)
        paths = list(self.metrics['reasoning_paths'].keys())
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                test_name = f'{paths[i]}_vs_{paths[j]}'
                tests[test_name] = self._compare_paths(paths[i], paths[j])

        # Calculate correlations
        tests['correlations'] = self._calculate_correlations()

        return tests

    def _calculate_performance_metrics(self,
                                       prediction: str,
                                       ground_truth: str) -> Dict[str, float]:
        """
        Calculate detailed performance metrics for a prediction.
        """
        return {
            'exact_match': float(prediction == ground_truth),
            'char_error_rate': self._calculate_char_error_rate(prediction, ground_truth),
            'prediction_length_ratio': float(len(prediction) / len(ground_truth))
        }

    def _calculate_path_efficiency(self,
                                   path_metrics: List[Dict[str, Any]]) -> float:
        """
        Calculate efficiency score for a reasoning path.
        """
        if not path_metrics:
            return 0.0

        processing_times = [m['processing_time'] for m in path_metrics]
        resource_usage = [sum(m[f'resource_{r}'] for r in ['cpu', 'memory', 'gpu'])
                          for m in path_metrics]

        # Normalize metrics
        norm_time = np.mean(processing_times) / np.max(processing_times) if np.max(processing_times) != 0 else 0
        norm_resource = np.mean(resource_usage) / np.max(resource_usage) if np.max(resource_usage) != 0 else 0

        # Calculate efficiency score (lower is better)
        return 1 - (0.5 * norm_time + 0.5 * norm_resource)

    def _save_metrics(self) -> None:
        """
        Save current metrics to disk.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"metrics_{self.experiment_name}_{timestamp}.json"

        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, default=str, indent=2)
            self.logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get real-time metrics for monitoring.
        """
        return {
            'current_query_count': self.query_count,
            'average_processing_time': float(np.mean(self.metrics['timing']['processing_times'])),
            'current_resource_usage': {
                resource: values[-1] if values else 0
                for resource, values in self.metrics['resource_usage'].items()
            },
            'path_distribution': {
                path: (len(metrics) / self.query_count) * 100
                for path, metrics in self.metrics['reasoning_paths'].items()
            }
        }

    def _calculate_efficiency_trends(self):
        """
        Calculate resource usage trends across queries.
        This implementation computes the difference between the last and the first recorded usage.
        """
        trends = {"cpu": 0.0, "memory": 0.0, "gpu": 0.0}
        # Retrieve query metrics from the metrics dictionary
        query_metrics_list = list(self.metrics['query_metrics'].values())
        if len(query_metrics_list) < 2:
            return trends
        first = query_metrics_list[0]
        last = query_metrics_list[-1]
        for resource in trends.keys():
            key = f"resource_{resource}"
            if key in first and key in last:
                trends[resource] = float(last[key] - first[key])
        return trends

    # Placeholder implementations for statistical tests and helper functions

    def _compare_paths(self, path1: str, path2: str) -> Dict[str, float]:
        """
        Compare two reasoning paths using a t-test.
        (Placeholder implementation)
        """
        # This is a placeholder. Replace with actual statistical test.
        return {"p_value": 0.05, "difference": 0.1}

    def _calculate_correlations(self) -> Dict[str, float]:
        """
        Calculate correlations among different metrics.
        (Placeholder implementation)
        """
        # This is a placeholder. Replace with actual correlation calculations.
        return {"correlation_coefficient": 0.5}

    def _calculate_char_error_rate(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate character error rate between prediction and ground truth.
        (Placeholder implementation using simple Levenshtein distance)
        """
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        distance = levenshtein(prediction, ground_truth)
        return distance / max(len(ground_truth), 1)
