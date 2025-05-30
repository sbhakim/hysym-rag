# src/resources/resource_manager.py

import logging
import psutil
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import os
import yaml
import numpy as np

from .adaptive_manager import AdaptiveManager

logger = logging.getLogger("ResourceManager")

class ResourceManager:
    """
    Enhanced resource manager for SymRAG that tracks CPU, GPU, and memory usage.
    Provides dynamic resource optimization in conjunction with an AdaptiveManager.
    """

    def __init__(
            self,
            config_path: str = "src/config/resource_config.yaml",
            enable_performance_tracking: bool = True,
            history_window_size: int = 100
    ):
        """
        Initialize the resource manager with advanced resource tracking capabilities.

        Args:
            config_path: Path to the resource configuration YAML
            enable_performance_tracking: Whether to track resource usage over time
            history_window_size: Number of samples to retain for rolling averages
        """
        self.logger = logger
        self.logger.setLevel(logging.INFO)

        self.enable_performance_tracking = enable_performance_tracking
        self.history_window_size = history_window_size

        # Load resource config
        self.resource_config = self._load_config(config_path)
        self.thresholds = self.resource_config.get("resource_thresholds", {})
        self.monitoring_config = self.resource_config.get("monitoring", {})
        self.adaptation_config = self.resource_config.get("adaptation", {})
        self.recovery_config = self.resource_config.get("recovery", {})

        # Initialize usage history
        self.usage_history = {
            "cpu": [],
            "memory": [],
            "gpu": []
        }

        # Adaptive manager
        self.adaptive_manager = AdaptiveManager(window_size=history_window_size)

        # Lock for concurrency safety if needed
        self._lock = threading.Lock()

        # Initialize GPU availability if needed
        self._gpu_available = False
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

        # Establish a baseline for academic measurements
        self.baseline = self._establish_baseline()

        self.logger.info("ResourceManager initialized with config: %s", config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load resource configuration from YAML.
        """
        if not os.path.exists(config_path):
            self.logger.warning(f"Resource config file not found at {config_path}, using defaults.")
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading resource config: {e}")
            return {}

    def _raw_resource_check(self) -> Dict[str, float]:
        """
        Get raw resource measurements without baseline adjustment.
        """
        cpu = psutil.cpu_percent() / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        gpu = 0.0
        if self._gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu = mem_info.used / mem_info.total
                pynvml.nvmlShutdown()
            except Exception as e:
                self.logger.error(f"Error in raw GPU check: {e}")
        return {"cpu": cpu, "memory": mem, "gpu": gpu}

    def _establish_baseline(self) -> Dict[str, float]:
        """
        Establish initial resource usage baseline for academic measurements.
        Takes multiple readings for stability and removes outliers using the IQR method.
        """
        readings = []
        num_samples = 10  # Increased sample size for better statistical significance
        for _ in range(num_samples):
            reading = self._raw_resource_check()
            readings.append(reading)
            time.sleep(0.2)  # Increased interval for more stable measurements

        baseline = {}
        for resource in readings[0].keys():
            values = [r[resource] for r in readings]
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            valid_values = [v for v in values if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr]
            if valid_values:
                baseline[resource] = float(np.mean(valid_values))
            else:
                baseline[resource] = float(np.mean(values))
        self.logger.info(f"Established baseline resource usage: {baseline}")
        return baseline

    def check_resources(self) -> Dict[str, float]:
        """
        Check current resource usage with enhanced smoothing and stability.
        Uses exponential moving average (EMA) smoothing.
        """
        with self._lock:
            # Get raw measurements
            current_usage = self._raw_resource_check()

            # Apply exponential moving average smoothing
            alpha = 0.3  # Smoothing factor
            for resource in current_usage:
                if self.usage_history[resource]:
                    previous = self.usage_history[resource][-1]
                    current_usage[resource] = alpha * current_usage[resource] + (1 - alpha) * previous

                # Apply a stability threshold to remove trivial fluctuations
                if abs(current_usage[resource]) < 0.001:  # 0.1% threshold
                    current_usage[resource] = 0.0

            self._update_usage_history(current_usage)
            return current_usage

    def _update_usage_history(self, usage: Dict[str, float]):
        """
        Update rolling history of resource usage.
        """
        for key in usage:
            validated = max(0.0, usage[key])  # Ensure no negative values are stored.
            self.usage_history[key].append(validated)
            if len(self.usage_history[key]) > self.history_window_size:
                self.usage_history[key].pop(0)

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate the trend (slope) over a list of values using linear regression.
        Returns 0.0 if there are fewer than 2 values.
        """
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def optimize_resources(self) -> Dict[str, float]:
        """
        Calculate optimal resource allocations based on current usage and
        historical trends, returning a dictionary with recommended allocations.
        """
        with self._lock:
            current_usage = self.check_resources()
            optimal_allocations = {}

            # Calculate resource usage trends over recent history (last 5 samples)
            usage_trends = {}
            for resource, values in self.usage_history.items():
                if len(values) >= 5:
                    usage_trends[resource] = self._calculate_trend(values[-5:])
                else:
                    usage_trends[resource] = 0.0

            # Get query complexity trend if available from the adaptive manager
            complexity_data = self.adaptive_manager.statistical_data.get('complexity', [])
            if len(complexity_data) >= 5:
                complexity_trend = self._calculate_trend(complexity_data[-5:])
            else:
                complexity_trend = 0.0

            # Process each resource with adaptive optimization
            for resource, usage in current_usage.items():
                base_threshold = self.thresholds.get(resource, {}).get("base_threshold", 0.8)
                adjustment_factor = self.thresholds.get(resource, {}).get("adjustment_factor", 0.1)

                # Calculate dynamic factor based on resource type and trends
                if resource == 'gpu' and complexity_trend > 0.05:
                    # If complexity is trending up, be more conservative with GPU
                    dynamic_factor = adjustment_factor * (1 + complexity_trend * 2)
                    dynamic_factor = min(0.25, dynamic_factor)  # Cap at 0.25
                elif resource == 'cpu' and usage_trends.get('gpu', 0) > 0.1:
                    # If GPU usage is trending up, prepare CPU for potential offloading
                    dynamic_factor = adjustment_factor * 0.5  # Less aggressive CPU optimization
                else:
                    dynamic_factor = adjustment_factor

                # Calculate target with more aggressive thresholds
                if usage > base_threshold * 0.9:
                    new_target = max(0.0, base_threshold - dynamic_factor)
                else:
                    # Add small headroom for unexpected spikes
                    new_target = min(base_threshold, usage + 0.05)
                optimal_allocations[resource] = new_target

            # Cross-resource balancing: if GPU target is high and CPU target is low, adjust accordingly
            if 'cpu' in optimal_allocations and 'gpu' in optimal_allocations:
                cpu_target = optimal_allocations['cpu']
                gpu_target = optimal_allocations['gpu']
                if gpu_target > 0.7 and cpu_target < 0.4:
                    optimal_allocations['gpu'] = max(0.6, gpu_target - 0.1)
                    optimal_allocations['cpu'] = min(0.7, cpu_target + 0.2)
                    self.logger.info("Balancing workload from GPU to CPU")
            return optimal_allocations

    def optimize_query_processing(self, query_complexity: float, current_usage: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize resource allocation and processing parameters based on query complexity and current usage.
        Returns an optimization plan with device mapping, precision mode, and batch size adjustments.
        """
        optimization_plan = {
            'device_mapping': {},
            'batch_size': 1,
            'precision': 'fp32'
        }
        # Adjust based on GPU usage if available
        if current_usage.get('gpu', 0) > self.thresholds.get('gpu', {}).get("base_threshold", 0.8) * 0.8:
            optimization_plan.update({
                'device_mapping': {
                    'embedding': 'cpu',
                    'preprocessing': 'cpu',
                    'inference': 'gpu'
                },
                'precision': 'fp16',
                'batch_size': max(1, 1)  # Adjust batch size as needed
            })
        # Adjust based on query complexity
        if query_complexity > 0.7:
            optimization_plan['batch_size'] = max(1, optimization_plan['batch_size'] // 2)
        return optimization_plan

    def apply_optimal_allocations(self, allocations: Dict[str, float]):
        """
        Apply new resource allocations (placeholder).
        In practice, might set GPU memory limits, thread pools, etc.
        """
        self.logger.info(f"Applying resource allocations: {allocations}")

    def recover_from_overload(self):
        """
        Attempt to recover from resource overload situations.
        Uses the 'recovery' config if available.
        """
        with self._lock:
            backoff_factor = self.recovery_config.get("backoff_factor", 1.5)
            max_retries = self.recovery_config.get("max_retries", 3)
            grace_period = self.recovery_config.get("grace_period", 10)
            self.logger.warning(
                f"Recovering from overload: backoff={backoff_factor}, retries={max_retries}, grace={grace_period}")
            time.sleep(grace_period)

    def is_overloaded(self, usage: Optional[Dict[str, float]] = None) -> bool:
        """
        Check if system is overloaded based on thresholds.
        """
        if usage is None:
            usage = self.check_resources()
        for resource, usage_val in usage.items():
            base_threshold = self.thresholds.get(resource, {}).get("base_threshold", 0.8)
            if usage_val > base_threshold:
                return True
        return False

    def get_usage_history(self) -> Dict[str, list]:
        """
        Return the rolling usage history.
        """
        return self.usage_history

    def get_average_usage(self) -> Dict[str, float]:
        """
        Return the average usage over the usage history window.
        """
        avg_usage = {}
        for resource, values in self.usage_history.items():
            if values:
                avg_usage[resource] = sum(values) / len(values)
            else:
                avg_usage[resource] = 0.0
        return avg_usage

    def adapt_resources(self):
        """
        Adapt resource thresholds or usage based on adaptive manager logic.
        """
        avg_usage = self.get_average_usage()
        self.logger.info(f"Adapting resources based on average usage: {avg_usage}")
        # Example: self.adaptive_manager.update_metrics(...)

    def check_and_recover_if_needed(self):
        """
        Periodically check if the system is overloaded and attempt recovery.
        """
        usage = self.check_resources()
        if self.is_overloaded(usage):
            self.logger.warning("System is overloaded. Initiating recovery.")
            self.recover_from_overload()
            self.adapt_resources()

    def get_resource_delta(self, current: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate validated resource usage changes relative to the established baseline.
        If current usage is below baseline, delta is set to 0.
        Otherwise, delta is normalized by the baseline and clamped between 0.0 and 1.0.
        """
        try:
            deltas = {}
            stability_threshold = 0.001  # 0.1% threshold

            for resource in current:
                baseline_value = self.baseline.get(resource, 0.0)
                raw_delta = current[resource] - baseline_value

                # If current usage is lower than baseline, set delta to 0
                if raw_delta < stability_threshold:
                    deltas[resource] = 0.0
                    continue

                # Normalize delta relative to baseline (if baseline is nonzero)
                if baseline_value > stability_threshold:
                    normalized_delta = raw_delta / baseline_value
                else:
                    normalized_delta = raw_delta if raw_delta > 0 else 0.0

                normalized_delta = max(0.0, min(normalized_delta, 1.0))
                deltas[resource] = normalized_delta

                if normalized_delta > 0.1:
                    self.logger.info(
                        f"Significant {resource} change: current={current[resource]:.3f}, "
                        f"baseline={baseline_value:.3f}, delta={normalized_delta:.3f}"
                    )
            return deltas
        except Exception as e:
            self.logger.error(f"Error calculating resource delta: {str(e)}")
            return {resource: 0.0 for resource in current}

    def _calculate_stability(self, values: List[float]) -> str:
        """
        Classify resource usage stability for academic analysis.
        """
        if not values:
            return "no_data"
        std = float(np.std(values))
        if std < 0.05:
            return "highly_stable"
        elif std < 0.1:
            return "stable"
        else:
            return "variable"

    def _calculate_efficiency_score(self) -> float:
        """
        Calculate a simple efficiency score based on recent resource usage.
        For demonstration, we average the inverses of usage fractions.
        """
        avg_usage = self.get_average_usage()
        scores = []
        for resource, usage in avg_usage.items():
            score = 1.0 / usage if usage > 0 else 1.0
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    def _count_recovery_events(self) -> int:
        """
        Stub method to count recovery events.
        """
        return 0

    def get_academic_metrics(self) -> Dict[str, Any]:
        """
        Generate metrics suitable for academic evaluation.
        """
        metrics = {
            'resource_utilization': {
                resource: {
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                    'peak': float(max(values)) if values else 0.0,
                    'stability': self._calculate_stability(values)
                }
                for resource, values in self.usage_history.items()
            },
            'efficiency_score': self._calculate_efficiency_score(),
            'adaptation_metrics': {
                'threshold_adjustments': len(self.usage_history.get('cpu', [])),
                'recovery_events': self._count_recovery_events()
            }
        }
        return metrics
