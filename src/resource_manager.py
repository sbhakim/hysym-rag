# src/resource_manager.py
import yaml
import psutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import pynvml

class ResourceManager:
    """
    Enhanced resource manager with adaptive thresholds, trend detection,
    predictive usage modeling, and optional neural performance tracking.
    """
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

    def check_resources(self):
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024**3)  # in GB
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu / 100.0
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / (1024**3)
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
        time_diffs = np.array([(item["timestamp"] - self.usage_history[0]["timestamp"]).total_seconds() for item in self.usage_history])
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

    def average_neural_inference_time(self):
        return np.mean(self.neural_perf_times) if self.neural_perf_times else 0.0

    def should_use_symbolic(self, query_complexity):
        usage = self.check_resources()
        trends = {res: self.calculate_trend(res) for res in ["cpu", "memory", "gpu"]}
        self.adjust_thresholds(trends)
        predicted_usage = {res: min(1.0, usage[res] * (1 + trends[res])) for res in ["cpu", "memory", "gpu"]}
        weights = {"cpu": 0.4, "memory": 0.3, "gpu": 0.3}
        risk_score = sum(predicted_usage[res] / self.current_thresholds[res] * weights[res] for res in predicted_usage)
        neural_perf_factor = 1.0 if self.average_neural_inference_time() >= 3.0 else 0.0
        overall_score = risk_score + neural_perf_factor
        if overall_score > 0.8 or query_complexity < 1.0:
            self.logger.info(f"Decision: symbolic (overall_score={overall_score:.2f}, query_complexity={query_complexity:.2f})")
            return True
        else:
            self.logger.info(f"Decision: neural (overall_score={overall_score:.2f}, query_complexity={query_complexity:.2f})")
            return False

    def monitor_resource_usage(self, inference_func):
        process = psutil.Process()
        start_mem = process.memory_info().rss
        start_time = time.time()
        inference_func()
        end_mem = process.memory_info().rss
        end_time = time.time()
        return {"memory_used_mb": round((end_mem - start_mem)/(1024**2), 4),
                "time_taken_s": round(end_time - start_time, 4)}
