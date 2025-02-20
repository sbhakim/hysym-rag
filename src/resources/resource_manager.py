# src/resources/resource_manager.py

import logging
import psutil
import time
from typing import Dict, Any, Optional
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

        # Initialize GPU usage if needed
        self._gpu_available = False
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

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

    def check_resources(self) -> Dict[str, float]:
        """
        Check current resource usage (CPU, memory, GPU).
        Returns usage as fraction of capacity (0.0 - 1.0).
        """
        with self._lock:
            cpu_usage = psutil.cpu_percent() / 100.0
            mem_usage = psutil.virtual_memory().percent / 100.0
            gpu_usage = 0.0

            if self._gpu_available:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_usage = mem_info.used / mem_info.total
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

            # Basic example: if usage is above threshold, reduce usage target
            optimal_allocations = {}
            for resource, usage in current_usage.items():
                base_threshold = self.thresholds.get(resource, {}).get("base_threshold", 0.8)
                adjustment_factor = self.thresholds.get(resource, {}).get("adjustment_factor", 0.1)

                # If usage is too high, lower the target; otherwise, keep it
                if usage > base_threshold:
                    new_target = max(0.0, base_threshold - adjustment_factor)
                else:
                    new_target = base_threshold

                optimal_allocations[resource] = new_target

            # Optionally incorporate adaptive manager logic
            # e.g. self.adaptive_manager.update_metrics("resource_optimization", ...)

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
        # This can be extended to incorporate advanced adaptive logic.
        # For example, we can use the usage history to detect patterns and
        # adjust thresholds dynamically.

        avg_usage = self.get_average_usage()
        self.logger.info(f"Adapting resources based on average usage: {avg_usage}")
        # Example usage: call self.adaptive_manager.update_metrics(...)

    def check_and_recover_if_needed(self):
        """
        Periodically check if the system is overloaded and attempt recovery.
        """
        usage = self.check_resources()
        if self.is_overloaded(usage):
            self.logger.warning("System is overloaded. Initiating recovery.")
            self.recover_from_overload()
            # Optionally adapt resources after recovery
            self.adapt_resources()
