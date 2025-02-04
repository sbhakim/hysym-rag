# src/resource_manager.py
import yaml
import psutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import pynvml
from collections import defaultdict

# Import the ResourceOptimizer
from src.resource_optimizer import ResourceOptimizer

# Import the PowerMonitor for energy tracking
from src.power_monitor import PowerMonitor

# For applying cgroup limits; ensure you have a suitable cgroups library installed.
# This is a placeholder; adjust the import and API according to your cgroups package.
import cgroups


class ResourceManager:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else Path("src/config/resource_config.yaml")
        self.load_config()
        self.usage_history = []
        self.window_size = self.monitoring.get("window_size", 10)
        self.last_adjustment = datetime.now()
        self.setup_logging()
        self.current_thresholds = {
            'cpu': self.resource_thresholds['cpu']['base_threshold'],
            'memory': self.resource_thresholds['memory']['base_threshold'],
            'gpu': self.resource_thresholds['gpu']['base_threshold']
        }
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.neural_perf_times = []
        # Initialize the ResourceOptimizer
        self.resource_optimizer = ResourceOptimizer()
        self.logger.info("ResourceManager: successfully initialized with advanced configuration.")

        # Initialize the PowerMonitor for energy tracking
        self.power_monitor = PowerMonitor()

    def setup_logging(self):
        logging.basicConfig(
            filename='logs/resource_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ResourceManager")

    def load_config(self):
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        self.resource_thresholds = cfg['resource_thresholds']
        self.monitoring = cfg['monitoring']
        self.adaptation = cfg['adaptation']

    def check_resources(self):
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 3)  # in GB
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu / 100.0
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / (1024 ** 3)
        usage = {"cpu": cpu_usage, "memory": memory_usage, "gpu": gpu_util, "gpu_mem": gpu_mem}
        self.update_usage_history(usage)
        return usage

    def update_usage_history(self, usage):
        self.usage_history.append({"usage": usage, "timestamp": datetime.now()})
        if len(self.usage_history) > self.window_size:
            self.usage_history.pop(0)

    def calculate_trend(self, resource):
        if len(self.usage_history) < 2:
            return 0.0
        weights = np.exp(np.linspace(-1, 0, len(self.usage_history)))
        weights /= weights.sum()
        values = np.array([item["usage"][resource] for item in self.usage_history])
        wma = np.sum(values * weights)
        time_diffs = np.array(
            [(item["timestamp"] - self.usage_history[0]["timestamp"]).total_seconds() for item in self.usage_history])
        slope = np.polyfit(time_diffs, values, 1)[0] if len(time_diffs) > 1 else 0.0
        diff_part = wma - values[0]
        combined = 0.7 * diff_part + 0.3 * slope
        return np.clip(combined, -1.0, 1.0)

    def adjust_thresholds(self, trends):
        if (datetime.now() - self.last_adjustment).total_seconds() < self.adaptation['cool_down_period']:
            return
        for res in ["cpu", "memory", "gpu"]:
            base_thresh = self.resource_thresholds[res]['base_threshold']
            adj_factor = self.resource_thresholds[res]['adjustment_factor']
            max_adj = self.adaptation['max_adjustment']
            adjustment = np.clip(trends[res] * adj_factor, -max_adj, max_adj)
            new_thresh = base_thresh - adjustment
            self.current_thresholds[res] = np.clip(new_thresh, base_thresh - max_adj, base_thresh + max_adj)
        self.last_adjustment = datetime.now()

    def monitor_energy(self):
        # Track energy consumption
        for component in ['gpu', 'cpu']:
            duration = self.get_usage_time(component)
            self.power_monitor.track(component, duration)

    def get_usage_time(self, component):
        # Simulate the time usage for the components (or replace with actual tracking method)
        # This could depend on the tracking mechanism you want, e.g., tracking GPU/CPU cycles.
        # Here we simply return an arbitrary value.
        return 1.0  # For demonstration, you would replace with actual tracking logic.

    def optimize_resources(self):
        """
        Uses the ResourceOptimizer to calculate optimal CPU and GPU allocations
        based on current resource usage.
        Returns:
            dict: Optimal allocations for 'cpu' and 'gpu'
        """
        usage = self.check_resources()
        cpu_usage = usage.get("cpu", 0)
        mem_usage = usage.get("memory", 0)
        gpu_usage = usage.get("gpu", 0)
        optimal_alloc = self.resource_optimizer.optimize(cpu_usage, mem_usage, gpu_usage)
        self.logger.info(f"Optimal allocations: {optimal_alloc}")
        # Apply the optimal allocations to the system
        self.apply_optimal_allocations(optimal_alloc)
        return optimal_alloc

    def apply_optimal_allocations(self, allocations):
        """
        Apply optimized resource limits using Linux cgroups.
        """
        try:
            # CPU allocation: convert fraction to percentage (e.g., 0.5 => 50%)
            cpu_limit = allocations['cpu'] * 100
            cgroup_cpu = cgroups.Cgroup('cpu')
            cgroup_cpu.set_cpu_limit(cpu_limit)
            self.logger.info(f"Applied CPU limit: {cpu_limit}%")

            # GPU memory allocation: set a limit based on optimal allocation for GPU memory.
            gpu_mem_limit = allocations.get('gpu_mem', 0.8) * (1024 ** 3)
            cgroup_gpu = cgroups.Cgroup('gpu')
            cgroup_gpu.set_memory_limit(gpu_mem_limit)
            self.logger.info(f"Applied GPU memory limit: {gpu_mem_limit / (1024 ** 3):.2f} GB")
        except Exception as e:
            self.logger.error(f"Error applying cgroup allocations: {str(e)}")


    def monitor_resource_usage(self, inference_func):
        """
        Monitors the resource usage (CPU, memory, GPU) while an inference function is executed.

        Args:
            inference_func (function): The function that will be executed for inference (e.g., processing a query).

        Returns:
            dict: A dictionary containing the resource usage information during inference.
        """
        start_cpu = psutil.cpu_percent(interval=0.1)
        start_mem = psutil.virtual_memory().percent
        start_gpu = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu

        start_time = time.time()
        inference_func()  # Execute the provided inference function
        end_time = time.time()

        end_cpu = psutil.cpu_percent(interval=0.1)
        end_mem = psutil.virtual_memory().percent
        end_gpu = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu

        # Calculate resource usage during inference
        cpu_usage = end_cpu - start_cpu
        memory_usage = end_mem - start_mem
        gpu_usage = end_gpu - start_gpu
        time_taken = end_time - start_time

        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage,
            "time_taken": time_taken
        }
