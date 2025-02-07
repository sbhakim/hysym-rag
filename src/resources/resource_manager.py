# src/resources/resource_manager.py

import os

import os
import yaml
import psutil
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import pynvml
from src.resources.resource_optimizer import ResourceOptimizer
from src.resources.power_monitor import PowerMonitor

logger = logging.getLogger("ResourceManager")


class EnergyAwareScheduler:
    """
    Schedules tasks based on real-time energy and resource usage.
    """

    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.task_queue = []  # Priority queue for tasks

    def schedule_next_task(self):
        if not self.task_queue:
            resources = self.resource_manager.check_resources()
            # Use GPU utilization from pynvml
            if resources.get("gpu", 0) > 0.8:
                return "symbolic"
            return "neural"
        return self.task_queue.pop(0)

    def schedule_task(self, task_type, energy_threshold=0.8):
        resources = self.resource_manager.check_resources()
        if task_type == "neural" and resources.get("gpu", 0) > energy_threshold:
            print("Energy usage too high. Falling back to symbolic reasoning.")
            return "symbolic"
        return task_type


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
        # Memory management parameters
        self.memory_high_water = 0.8  # Use up to 80% of system RAM
        self.gpu_memory_fraction = 0.8  # Use up to 80% of GPU memory
        self.peak_memory = 0
        self.memory_readings = []

        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.neural_perf_times = []
        self.resource_optimizer = ResourceOptimizer()
        self.logger.info("ResourceManager: successfully initialized with advanced configuration.")
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

        # Update usage history and peak memory.
        self.peak_memory = max(self.peak_memory, memory_usage)
        self.memory_readings.append(memory_usage)
        if len(self.memory_readings) > self.window_size:
            self.memory_readings.pop(0)

        # Log GPU memory usage summary (if CUDA is available).
        if torch.cuda.is_available():
            self.logger.info("GPU Memory Summary:\n" + torch.cuda.memory_summary(device="cuda"))

        total_mem = psutil.virtual_memory().total / (1024 ** 3)
        if memory_usage > (total_mem * self.memory_high_water):
            self.emergency_cleanup()

        self.update_usage_history(usage)
        return usage

    def update_usage_history(self, usage):
        self.usage_history.append({"usage": usage, "timestamp": datetime.now()})
        if len(self.usage_history) > self.window_size:
            self.usage_history.pop(0)

    def check_memory_critical(self):
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 3)
        total_mem = psutil.virtual_memory().total / (1024 ** 3)
        return memory_usage > (total_mem * self.memory_high_water)

    def emergency_cleanup(self):
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("ResourceManager: Emergency cleanup performed due to high memory usage.")

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
        for component in ['gpu', 'cpu']:
            duration = self.get_usage_time(component)
            self.power_monitor.track(component, duration)

    def get_usage_time(self, component):
        return 1.0

    def optimize_resources(self):
        usage = self.check_resources()
        cpu_usage = usage.get("cpu", 0)
        mem_usage = usage.get("memory", 0)
        gpu_usage = usage.get("gpu", 0)
        optimal_alloc = self.resource_optimizer.optimize(cpu_usage, mem_usage, gpu_usage)
        # Clamp GPU allocation to a small positive value if it computes as 0 or negative.
        optimal_alloc['gpu'] = max(0.1, optimal_alloc.get('gpu', 0))
        self.logger.info(f"Optimal allocations: {optimal_alloc}")
        self.apply_optimal_allocations(optimal_alloc)
        return optimal_alloc

    def apply_optimal_allocations(self, allocations):
        try:
            cpu_limit = allocations['cpu'] * 100
            self._apply_soft_limits(allocations)
            self.logger.info(f"Applied soft CPU limit: {cpu_limit}%")
            self.logger.info("Skipping GPU cgroup limits; running in regular mode.")
        except Exception as e:
            self.logger.error(f"Error applying resource allocations: {str(e)}")

    def has_cgroup_access(self):
        return False

    def _apply_soft_limits(self, allocations):
        import resource
        if 'cpu' in allocations:
            cpu_limit = int(allocations['cpu'] * 100)
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
                self.logger.info(f"Applied soft CPU limit: {cpu_limit}%")
            except Exception as e:
                self.logger.error(f"Error applying soft CPU limit: {str(e)}")

    def monitor_resource_usage(self, inference_func):
        start_cpu = psutil.cpu_percent(interval=0.1)
        start_mem = psutil.virtual_memory().percent
        start_gpu = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
        start_time = time.time()
        inference_func()
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=0.1)
        end_mem = psutil.virtual_memory().percent
        end_gpu = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
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
