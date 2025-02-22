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
    Enhanced resource manager for HySym-RAG that tracks CPU, GPU, and memory usage.
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
        Takes multiple readings for stability.
        """
        readings = []
        for _ in range(5):  # Take 5 readings
            readings.append(self._raw_resource_check())
            time.sleep(0.1)
        baseline = {resource: sum(r[resource] for r in readings) / len(readings)
                    for resource in readings[0]}
        self.logger.info(f"Established baseline resource usage: {baseline}")
        return baseline

    def check_resources(self) -> Dict[str, float]:
        """
        Check current resource usage (CPU, memory, GPU).
        Returns usage as fraction of capacity (0.0 - 1.0).
        """
        with self._lock:
            # Ensure non-negative values using max(0.0, ...)
            cpu_usage = max(0.0, psutil.cpu_percent(interval=0.1) / 100.0)
            mem_usage = max(0.0, psutil.virtual_memory().percent / 100.0)
            gpu_usage = 0.0

            if self._gpu_available:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_usage = max(0.0, mem_info.used / mem_info.total)
                    pynvml.nvmlShutdown()
                except Exception as e:
                    self.logger.error(f"Error getting GPU usage: {e}")

            resource_usage = {
                "cpu": cpu_usage,
                "memory": mem_usage,
                "gpu": gpu_usage
            }

            # Update usage history
            self._update_usage_history(resource_usage)
            return resource_usage

    def _update_usage_history(self, usage: Dict[str, float]):
        """
        Update rolling history of resource usage.
        """
        for key in usage:
            self.usage_history[key].append(usage[key])
            if len(self.usage_history[key]) > self.history_window_size:
                self.usage_history[key].pop(0)

    def optimize_resources(self) -> Dict[str, float]:
        """
        Calculate optimal resource allocations based on current usage and
        historical trends, returning a dictionary with recommended allocations.
        """
        with self._lock:
            current_usage = self.check_resources()
            optimal_allocations = {}
            for resource, usage in current_usage.items():
                base_threshold = self.thresholds.get(resource, {}).get("base_threshold", 0.8)
                adjustment_factor = self.thresholds.get(resource, {}).get("adjustment_factor", 0.1)
                if usage > base_threshold:
                    new_target = max(0.0, base_threshold - adjustment_factor)
                else:
                    new_target = base_threshold
                optimal_allocations[resource] = new_target
            return optimal_allocations

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
            self.logger.warning(f"Recovering from overload: backoff={backoff_factor}, retries={max_retries}, grace={grace_period}")
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
        Calculate resource usage changes relative to the established baseline.
        Returns absolute differences.
        """
        return {
            resource: abs(current[resource] - self.baseline.get(resource, 0.0))
            for resource in current
        }

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
            # Avoid division by zero; assume perfect efficiency if usage is 0
            score = 1.0 / usage if usage > 0 else 1.0
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    def _count_recovery_events(self) -> int:
        """
        Stub method to count recovery events.
        In a complete implementation, this would track the number of times recovery has been triggered.
        """
        # For now, return 0 as a placeholder.
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
