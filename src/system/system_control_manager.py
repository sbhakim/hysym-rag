# src/system/system_control_manager.py

import logging
import time
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
import torch
import numpy as np
from collections import defaultdict
from src.utils.metrics_collector import MetricsCollector

class UnifiedResponseAggregator:
    """
    Aggregator for academic evaluation responses with detailed reasoning.
    """
    def __init__(self, include_explanations: bool = True):
        self.include_explanations = include_explanations

    def format_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.include_explanations:
            data.setdefault("explanation", self._generate_detailed_explanation(data))
        return data

    def _generate_detailed_explanation(self, data: Dict[str, Any]) -> str:
        """Generate detailed reasoning explanation for academic analysis."""
        parts = []
        if 'reasoning_path' in data:
            parts.append(f"Reasoning Approach: {data['reasoning_path']}")
        if 'processing_time' in data:
            parts.append(f"Processing Time: {data['processing_time']:.3f}s")
        if 'resource_usage' in data:
            parts.append("Resource Utilization:")
            for resource, usage in data['resource_usage'].items():
                parts.append(f"- {resource}: {usage*100:.1f}%")
        return " | ".join(parts) if parts else "No additional explanations provided."

class SystemControlManager:
    """
    Enhanced SystemControlManager for academic evaluation of HySym-RAG.
    Focuses on reproducible metrics, detailed analysis, and academic logging.
    """
    def __init__(
            self,
            hybrid_integrator,
            resource_manager,
            aggregator,
            metrics_collector: Optional[MetricsCollector] = None,
            error_retry_limit: int = 2,
            max_query_time: float = 10.0,
            performance_window: int = 100
    ):
        self.hybrid_integrator = hybrid_integrator
        self.resource_manager = resource_manager
        self.aggregator = aggregator
        self.error_retry_limit = error_retry_limit
        self.max_query_time = max_query_time
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger("SystemControlManager")
        self.logger.setLevel(logging.INFO)

        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'path_performance': defaultdict(list),
            'resource_efficiency': defaultdict(list),
            'reasoning_quality': defaultdict(list)
        }
        self.path_history = []
        self.adaptive_thresholds = {
            'complexity_threshold': 0.8,
            'resource_pressure_threshold': 0.8,
            'efficiency_threshold': 0.6
        }

    def process_query_with_fallback(
            self,
            query: str,
            context: str,
            max_retries: Optional[int] = None,
            forced_path: Optional[str] = None,
            query_complexity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process query with enhanced error handling and metrics collection.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        start_time = time.time()
        retries = 0
        max_retries = max_retries or self.error_retry_limit

        # Record initial resource state
        initial_resources = self.resource_manager.check_resources()

        while retries <= max_retries:
            try:
                resource_status = self.resource_manager.check_resources()
                if not self._validate_resources(resource_status):
                    self.logger.warning("Resource limits exceeded, optimizing allocation...")
                    self._optimize_resources()

                result, reasoning_path = self._timed_process_query(
                    query,
                    context,
                    forced_path,
                    query_complexity
                )

                processing_time = time.time() - start_time
                resource_delta = self._calculate_resource_delta(
                    initial_resources,
                    self.resource_manager.check_resources()
                )

                self._update_metrics(start_time, True, reasoning_path)

                # Collect academic metrics
                self.metrics_collector.collect_query_metrics(
                    query=query,
                    prediction=result if not isinstance(result, tuple) else result[0],
                    ground_truth=None,  # Ground truth can be added if available
                    reasoning_path=reasoning_path,
                    processing_time=processing_time,
                    resource_usage=resource_delta,
                    complexity_score=query_complexity or 0.0
                )

                return self.aggregator.format_response({
                    'result': result,
                    'processing_time': processing_time,
                    'resource_usage': resource_delta,
                    'reasoning_path': reasoning_path,
                    'retries': retries
                })

            except TimeoutError as e:
                self.logger.error(f"Query processing timeout: {str(e)}")
                retries += 1
                if retries > max_retries:
                    return self._handle_final_failure(start_time, "Timeout")
                continue

            except Exception as e:
                self.logger.error(f"Error processing query: {str(e)}")
                retries += 1
                if retries > max_retries:
                    return self._handle_final_failure(start_time, str(e))
                continue

    def _timed_process_query(
            self,
            query: str,
            context: str,
            forced_path: Optional[str] = None,
            query_complexity: Optional[float] = None
    ) -> Tuple[Any, str]:
        """
        Process query with enhanced timing and path selection logging.
        """
        start = time.time()
        if forced_path:
            optimal_path = forced_path
            self.logger.info(f"Forced reasoning path: {optimal_path}")
        else:
            if query_complexity is None and hasattr(self.hybrid_integrator, "query_expander"):
                query_complexity = self.hybrid_integrator.query_expander.get_query_complexity(query)
                self.logger.info(f"Computed query complexity: {query_complexity:.4f}")
            elif query_complexity is None:
                query_complexity = 0.8  # Default fallback

            optimal_path = self._determine_optimal_path(
                query_complexity,
                self.resource_manager.check_resources()
            )
            self.logger.info(f"Selected reasoning path: {optimal_path}")

        result = self._execute_processing_path(
            optimal_path,
            query,
            context,
            query_complexity
        )

        processing_time = time.time() - start
        self._update_path_stats(optimal_path, processing_time)
        return result, optimal_path

    def _determine_optimal_path(
            self,
            query_complexity: float,
            resource_status: Dict[str, float]
    ) -> str:
        """
        Determine optimal processing path with enhanced decision logging.
        """
        resource_pressure = max(
            resource_status.get('cpu', 0),
            resource_status.get('memory', 0),
            resource_status.get('gpu', 0)
        )

        decision_factors = {
            'complexity': query_complexity,
            'resource_pressure': resource_pressure
        }

        if query_complexity < self.adaptive_thresholds['complexity_threshold']:
            path = "symbolic"
            reason = "Low query complexity favors symbolic reasoning"
        elif resource_pressure > self.adaptive_thresholds['resource_pressure_threshold']:
            path = "symbolic"
            reason = "High resource pressure favors symbolic path"
        else:
            path = "hybrid"
            reason = "Balanced complexity and resource usage favor hybrid approach"

        self.logger.info(f"Path Selection Factors: {decision_factors}")
        self.logger.info(f"Selected Path: {path} - Reason: {reason}")

        self.path_history.append({
            'timestamp': datetime.now().isoformat(),
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
            query_complexity: float
    ) -> Any:
        """
        Execute query processing along the selected path.
        """
        if path == "symbolic":
            return self.hybrid_integrator.symbolic_reasoner.process_query(query)
        elif path == "neural":
            return self.hybrid_integrator.neural.retrieve_answer(
                context,
                query,
                query_complexity=query_complexity
            )
        else:  # hybrid path
            return self.hybrid_integrator.process_query(
                query,
                context,
                query_complexity=query_complexity
            )

    def _optimize_resources(self):
        """
        Optimize resource allocation based on current system state.
        """
        current_allocation = self.resource_manager.check_resources()
        optimal_allocation = self.resource_manager.optimize_resources()
        if self._should_reallocate(current_allocation, optimal_allocation):
            self.resource_manager.apply_optimal_allocations(optimal_allocation)
            self.logger.info("Applied new resource allocation")

    def _update_metrics(
            self,
            start_time: float,
            success: bool,
            reasoning_type: str
    ):
        """
        Update system performance metrics.
        """
        try:
            response_time = time.time() - start_time
            self.performance_metrics['total_queries'] += 1
            if success:
                self.performance_metrics['successful_queries'] += 1
            else:
                self.performance_metrics['failed_queries'] += 1

            total = self.performance_metrics['total_queries']
            current_avg = self.performance_metrics['avg_response_time']
            self.performance_metrics['avg_response_time'] = ((current_avg * (total - 1)) + response_time) / total
            self.performance_metrics['path_performance'][reasoning_type].append(response_time)
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system performance metrics for academic analysis.
        """
        total = max(1, self.performance_metrics['total_queries'])
        basic_metrics = {
            'total_queries': total,
            'success_rate': (self.performance_metrics['successful_queries'] / total) * 100,
            'avg_response_time': self.performance_metrics['avg_response_time'],
            'error_rate': (self.performance_metrics['failed_queries'] / total) * 100,
            'path_distribution': self._calculate_path_distribution(),
            'resource_efficiency': self._calculate_resource_efficiency()
        }
        return basic_metrics

    def _calculate_path_distribution(self) -> Dict[str, float]:
        """
        Calculate detailed path distribution statistics.
        """
        total_queries = len(self.path_history)
        if not total_queries:
            return {}
        distribution = defaultdict(int)
        for record in self.path_history:
            distribution[record['path']] += 1
        return {path: (count / total_queries) * 100 for path, count in distribution.items()}

    def _calculate_resource_efficiency(self) -> Dict[str, Any]:
        """
        Calculate comprehensive resource efficiency metrics.
        """
        if not self.performance_metrics['resource_efficiency']:
            return {}
        efficiency_metrics = {}
        for resource, values in self.performance_metrics['resource_efficiency'].items():
            if values:
                avg_usage = np.mean(values)
                peak_usage = max(values)
                efficiency_metrics[resource] = {
                    'average_usage': avg_usage,
                    'peak_usage': peak_usage,
                    'efficiency_score': 1 - (avg_usage / peak_usage) if peak_usage else 0
                }
        return efficiency_metrics

    def _calculate_resource_delta(
            self,
            initial_resources: Dict[str, float],
            final_resources: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate difference in resource usage between two states.
        """
        return {key: final_resources.get(key, 0.0) - initial_resources.get(key, 0.0)
                for key in final_resources}

    def _validate_resources(self, resource_status: Dict[str, float]) -> bool:
        """
        Check if resources are within acceptable limits.
        """
        if resource_status.get("cpu", 0) > 0.95:
            return False
        if resource_status.get("memory", 0) > 0.95:
            return False
        if resource_status.get("gpu", 0) > 0.99:
            return False
        return True

    def _update_path_stats(self, path: str, duration: float):
        """
        Update path distribution and timing statistics.
        """
        if path not in self.performance_metrics['path_performance']:
            self.performance_metrics['path_performance'][path] = []
        self.performance_metrics['path_performance'][path].append(duration)

    def _should_reallocate(
            self,
            current_allocation: Dict[str, float],
            optimal_allocation: Dict[str, float]
    ) -> bool:
        """
        Check if there's a significant difference between current and optimal allocations.
        """
        threshold = 0.05
        for resource in current_allocation:
            if abs(optimal_allocation.get(resource, 0.0) - current_allocation[resource]) > threshold:
                return True
        return False

    def _handle_final_failure(self, start_time: float, reason: str) -> Dict[str, Any]:
        """
        Return a final error response after maximum retries or fatal exception.
        """
        self._update_metrics(start_time, False, "fallback")
        return {
            "error": reason,
            "status": "failed"
        }

    def get_reasoning_path_stats(self) -> Dict[str, Any]:
        """
        Get statistics on reasoning path distribution.
        """
        path_counts = defaultdict(lambda: {"count": 0})
        for record in self.path_history: # Assuming self.path_history logs path choices
            path_counts[record['path']]["count"] += 1

        total_queries = self.performance_metrics['total_queries']
        path_stats = {}
        for path, data in path_counts.items():
            path_stats[path] = {
                "count": data["count"],
                "percentage": (data["count"] / total_queries) * 100 if total_queries > 0 else 0
            }
        return path_stats
