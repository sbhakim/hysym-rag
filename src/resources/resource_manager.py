# src/resources/resource_manager.py

import yaml
import psutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pynvml

# Import the ResourceOptimizer
from src.resources.resource_optimizer import ResourceOptimizer
# Import the PowerMonitor for energy tracking
from src.resources.power_monitor import PowerMonitor
# Import the new AdaptiveManager for learning resource thresholds
from src.resources.adaptive_manager import AdaptiveManager

# For applying cgroup limits; in development, we disable these calls.
import cgroups


class ResourceManager:
    def __init__(self, config_path=None, enable_performance_tracking=True, history_window_size=100):
        """
        Initialize ResourceManager with advanced performance tracking and adaptive resource management.

        Args:
            config_path: Path to resource configuration file
            enable_performance_tracking: Flag to enable performance history tracking
            history_window_size: Number of recent queries to maintain in history
        """
        # Basic initialization
        self.config_path = Path(config_path) if config_path else Path("src/config/resource_config.yaml")
        self.load_config()
        self.setup_logging()

        # Performance tracking settings
        self.enable_performance_tracking = enable_performance_tracking
        self.history_window_size = history_window_size
        self.usage_history = []
        self.window_size = self.monitoring.get("window_size", 10)
        self.last_adjustment = datetime.now()

        # Resource thresholds with dynamic adjustment capabilities
        self.current_thresholds = {
            'cpu': self.resource_thresholds['cpu']['base_threshold'],
            'memory': self.resource_thresholds['memory']['base_threshold'],
            'gpu': self.resource_thresholds['gpu']['base_threshold']
        }

        # Initialize GPU monitoring
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.neural_perf_times = []

        # Performance history tracking for each reasoning type
        self.performance_history = {
            'symbolic': {
                'latency': [],
                'accuracy': [],
                'resource_usage': [],
                'success_count': 0
            },
            'neural': {
                'latency': [],
                'accuracy': [],
                'resource_usage': [],
                'success_count': 0
            },
            'hybrid': {
                'latency': [],
                'accuracy': [],
                'resource_usage': [],
                'success_count': 0
            }
        }

        # Adaptive thresholds for performance optimization
        self.adaptive_thresholds = {
            'latency_threshold': 1.0,  # seconds
            'accuracy_threshold': 0.7,
            'resource_efficiency': 0.8,
            'min_success_rate': 0.6,
            'max_resource_usage': 0.9
        }

        # Resource optimization components
        self.resource_optimizer = ResourceOptimizer()
        self.power_monitor = PowerMonitor()
        # Initialize the adaptive manager
        self.adaptive_manager = AdaptiveManager()

        # Performance tracking metrics
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0

        self.logger.info("ResourceManager: successfully initialized with advanced configuration.")

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

    def check_resources(self, query_type=None):
        """
        Enhanced resource checking with adaptive thresholds.
        Optionally accepts a query_type to use learned thresholds.
        """
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 3)  # in GB
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu / 100.0
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / (1024 ** 3)
        usage = {"cpu": cpu_usage, "memory": memory_usage, "gpu": gpu_util, "gpu_mem": gpu_mem}
        self.update_usage_history(usage)

        if query_type:
            thresholds = self.adaptive_manager.get_thresholds(query_type)
            usage['thresholds'] = thresholds

            # Check if we're approaching thresholds (90% of each)
            for resource, value in usage.items():
                if resource in thresholds:
                    if value > thresholds[resource] * 0.9:
                        self.logger.warning(
                            f"Resource {resource} approaching threshold for query type {query_type}"
                        )

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
        # For demonstration, simply return an arbitrary value.
        return 1.0

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
        # For development, disable cgroup application by simply not applying any limits.
        # self.apply_optimal_allocations(optimal_alloc)
        self.logger.info("Cgroup allocation disabled for development.")
        return optimal_alloc

    def apply_optimal_allocations(self, allocations):
        """
        Apply optimized resource limits using Linux cgroups.
        In production, ensure your system has proper cgroup hierarchies.
        For development, you can safely disable this function.
        """
        try:
            # Uncomment the following lines in production when cgroup hierarchies are available.
            # cpu_limit = allocations['cpu'] * 100
            # cgroup_cpu = cgroups.Cgroup('cpu')
            # cgroup_cpu.set_cpu_limit(cpu_limit)
            # self.logger.info(f"Applied CPU limit: {cpu_limit}%")
            #
            # gpu_mem_limit = allocations.get('gpu_mem', 0.8) * (1024 ** 3)
            # cgroup_gpu = cgroups.Cgroup('gpu')
            # cgroup_gpu.set_memory_limit(gpu_mem_limit)
            # self.logger.info(f"Applied GPU memory limit: {gpu_mem_limit / (1024 ** 3):.2f} GB")
            pass
        except Exception as e:
            self.logger.error(f"Error applying cgroup allocations: {str(e)}")

    def monitor_resource_usage(self, inference_func):
        """
        Monitors the resource usage while an inference function is executed.
        """
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

    def update_performance_metrics(self, reasoning_type: str, metrics: dict):
        """
        Update performance history with new metrics from query execution.

        Args:
            reasoning_type: Type of reasoning used (symbolic, neural, or hybrid)
            metrics: Dictionary containing performance metrics
        """
        if not self.enable_performance_tracking:
            return

        history = self.performance_history.get(reasoning_type)
        if not history:
            self.logger.warning(f"Unknown reasoning type: {reasoning_type}")
            return

        # Update performance history
        history['latency'].append(metrics.get('latency', 0))
        history['accuracy'].append(metrics.get('accuracy', 0))
        history['resource_usage'].append(metrics.get('resource_usage', {}))

        # Maintain history window size
        if len(history['latency']) > self.history_window_size:
            history['latency'] = history['latency'][-self.history_window_size:]
            history['accuracy'] = history['accuracy'][-self.history_window_size:]
            history['resource_usage'] = history['resource_usage'][-self.history_window_size:]

        # Update success counts
        if metrics.get('success', True):
            history['success_count'] += 1
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        self.total_queries += 1

    def calculate_efficiency_score(self, reasoning_type: str) -> float:
        """
        Calculate efficiency score for a reasoning path based on historical performance.

        Args:
            reasoning_type: Type of reasoning to evaluate

        Returns:
            float: Efficiency score (0-1)
        """
        history = self.performance_history[reasoning_type]
        if not history['latency']:
            return 0.5

        # Calculate recent performance metrics
        recent_latency = np.mean(history['latency'][-10:])
        recent_accuracy = np.mean(history['accuracy'][-10:])
        recent_resource = np.mean([
            sum(usage.values()) / len(usage)
            for usage in history['resource_usage'][-10:]
        ])

        # Calculate score components
        latency_score = 1.0 / (1.0 + recent_latency)
        success_rate = history['success_count'] / max(1, len(history['latency']))

        # Combine metrics into final score
        efficiency_score = (
                0.4 * recent_accuracy +
                0.3 * latency_score +
                0.2 * (1.0 - recent_resource) +
                0.1 * success_rate
        )

        return np.clip(efficiency_score, 0, 1)

    def _get_default_path(self, query_complexity: float) -> str:
        """
        Provides a default reasoning path based on query complexity.
        """
        if query_complexity < 0.5:
            return "symbolic"
        elif query_complexity < 1.0:
            return "hybrid"
        else:
            return "neural"

    def get_optimal_reasoning_path(self, query_complexity: float) -> str:
        """
        Determine the optimal reasoning path based on historical performance.

        Args:
            query_complexity: Complexity score of the query (0-1)

        Returns:
            str: Recommended reasoning path (symbolic, neural, or hybrid)
        """
        if not self.enable_performance_tracking or self.total_queries < 5:
            # Use default complexity-based routing for initial queries
            return self._get_default_path(query_complexity)

        # Calculate efficiency scores for each path
        scores = {}
        for path_type in ['symbolic', 'neural', 'hybrid']:
            scores[path_type] = self.calculate_efficiency_score(path_type)

        # Consider query complexity in decision
        if query_complexity < 0.4:
            return 'symbolic' if scores['symbolic'] > 0.4 else 'hybrid'
        elif query_complexity < 0.7:
            best_path = max(scores.items(), key=lambda x: x[1])[0]
            return best_path if scores[best_path] > 0.6 else 'hybrid'
        else:
            return 'neural' if scores['neural'] > 0.5 else 'hybrid'

    def optimize_resources(self):
        """Enhanced resource optimization with adaptive thresholds."""
        current_usage = self.check_resources()

        # Calculate trending efficiency
        symbolic_efficiency = self.calculate_efficiency_score('symbolic')
        neural_efficiency = self.calculate_efficiency_score('neural')

        # Adjust resource allocation based on efficiency scores
        optimal_allocation = {
            'cpu': max(0, 1 - current_usage['cpu']),
            'gpu': max(0, 1 - current_usage['gpu']),
            'gpu_mem': min(0.9, 0.8 + 0.1 * neural_efficiency)
        }

        self.logger.info(f"Optimal allocations: {optimal_allocation}")
        # For development, cgroup allocation is disabled.
        # self.apply_optimal_allocations(optimal_allocation)
        self.logger.info("Cgroup allocation disabled for development.")
        return optimal_allocation
