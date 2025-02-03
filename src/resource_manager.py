# src/resource_manager.py
import yaml
import psutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

class ResourceManager:
    """
    Enhanced resource manager with adaptive thresholds, trend detection, and predictive usage modeling.
    Loads config from YAML, tracks usage, adjusts thresholds if usage is rising.
    """

    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else Path("src/config/resource_config.yaml")
        self.load_config()

        # Usage history for trend analysis
        self.usage_history = []
        self.window_size = self.monitoring.get("window_size", 10)
        self.last_adjustment = datetime.now()

        # Setup logging
        self.setup_logging()

        # Initialize current thresholds
        self.current_thresholds = {
            'cpu': self.resource_thresholds['cpu']['base_threshold'],
            'memory': self.resource_thresholds['memory']['base_threshold'],
            'gpu': self.resource_thresholds['gpu']['base_threshold']
        }
        print("ResourceManager initialized with advanced configuration.")

    def setup_logging(self):
        logging.basicConfig(
            filename='logs/resource_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ResourceManager")

    def load_config(self):
        """
        Load resource thresholds and adaptation from the YAML config.
        """
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        self.resource_thresholds = cfg['resource_thresholds']
        self.monitoring = cfg['monitoring']
        self.adaptation = cfg['adaptation']

    def check_resources(self):
        """
        Return current usage of CPU/memory plus a placeholder GPU usage (if not monitored).
        """
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        mem = psutil.virtual_memory()
        memory_usage = mem.percent / 100.0

        # Placeholder for GPU usage (0.0 if not monitored)
        gpu_usage = 0.0

        usage = {
            "cpu": cpu_usage,
            "memory": memory_usage,
            "gpu": gpu_usage
        }
        self.update_usage_history(usage)
        return usage

    def update_usage_history(self, usage):
        """
        Keep track of recent usage samples, discarding old entries beyond self.window_size.
        """
        self.usage_history.append({
            "usage": usage,
            "timestamp": datetime.now()
        })
        if len(self.usage_history) > self.window_size:
            self.usage_history.pop(0)

    def calculate_trend(self, resource):
        """
        Enhanced trend calculation using a weighted moving average + slope estimate.
        - We use an exponential weights array, apply it to resource usage values,
          then do a simple linear fit to estimate slope.
        - Combine WMA difference and slope for the final 'trend' measure, clamped [-1,1].
        """
        if len(self.usage_history) < 2:
            return 0.0

        # Weighted moving average using exponential weights
        weights = np.exp(np.linspace(-1, 0, len(self.usage_history)))
        weights /= weights.sum()

        values = np.array([item["usage"][resource] for item in self.usage_history])
        wma = np.sum(values * weights)

        # Estimate slope via polyfit over time
        time_diffs = np.array([
            (item["timestamp"] - self.usage_history[0]["timestamp"]).total_seconds()
            for item in self.usage_history
        ])
        if len(time_diffs) > 1:
            slope, _ = np.polyfit(time_diffs, values, 1)
        else:
            slope = 0.0

        # Combine immediate difference with slope
        diff_part = wma - values[0]
        combined = 0.7 * diff_part + 0.3 * slope
        return np.clip(combined, -1.0, 1.0)

    def adjust_thresholds(self, trends):
        """
        Dynamically adjust thresholds based on usage trends and a cool-down.
        """
        if (datetime.now() - self.last_adjustment).total_seconds() < self.adaptation['cool_down_period']:
            return  # skip adjustments until cool-down passes

        for res in ["cpu", "memory", "gpu"]:
            base_thresh = self.resource_thresholds[res]['base_threshold']
            adj_factor = self.resource_thresholds[res]['adjustment_factor']
            max_adj = self.adaptation['max_adjustment']

            # Trend-based adjustment
            adjustment = trends[res] * adj_factor
            adjustment = np.clip(adjustment, -max_adj, max_adj)

            new_thresh = base_thresh - adjustment
            # Ensure we don't exceed some range
            self.current_thresholds[res] = np.clip(
                new_thresh,
                base_thresh - max_adj,
                base_thresh + max_adj
            )

        self.last_adjustment = datetime.now()

    def should_use_symbolic(self, query_complexity):
        """
        Enhanced decision making with predictive usage modeling.
        1. Check current usage and resource trends.
        2. Predict future usage based on current usage * (1 + trend).
        3. Compute a 'risk_score' that compares predicted usage to thresholds.
        4. If risk is high, or query complexity is low, prefer symbolic.
        """
        usage = self.check_resources()
        trends = {res: self.calculate_trend(res) for res in ["cpu", "memory", "gpu"]}

        # Possibly adjust thresholds if usage is trending
        self.adjust_thresholds(trends)

        # Predict future usage
        predicted_usage = {
            res: min(1.0, usage[res] * (1 + trends[res]))
            for res in usage
        }

        # Calculate a simple risk score (ratio of predicted usage to current threshold)
        risk_score = 0.0
        for res in predicted_usage:
            # Avoid dividing by zero
            if self.current_thresholds[res] > 0:
                ratio = predicted_usage[res] / self.current_thresholds[res]
            else:
                ratio = 0
            risk_score += ratio
        risk_score /= len(predicted_usage)

        # Example thresholds for deciding to go symbolic:
        # 1) Very high predicted usage
        # 2) Low query complexity (since symbolic is cheap)
        # 3) Medium complexity + medium risk
        if risk_score > 0.8:
            self.logger.info(f"High risk score {risk_score:.2f}: fallback to symbolic.")
            return True

        if query_complexity < 1.0:
            self.logger.info(f"Low complexity {query_complexity:.2f}: fallback to symbolic.")
            return True

        if query_complexity < 2.0 and risk_score > 0.6:
            self.logger.info(
                f"Moderate complexity {query_complexity:.2f} + moderate risk {risk_score:.2f}: fallback to symbolic."
            )
            return True

        # Otherwise, proceed with neural
        return False

    def monitor_resource_usage(self, inference_func):
        """
        Track memory/time usage around inference. Returns a dict with 'memory_used_mb' and 'time_taken_s'.
        """
        start_mem = psutil.virtual_memory().used
        start_time = time.time()

        # Perform the inference or processing
        inference_func()

        end_mem = psutil.virtual_memory().used
        end_time = time.time()

        return {
            "memory_used_mb": round((end_mem - start_mem)/(1024**2), 4),
            "time_taken_s": round(end_time - start_time, 4)
        }
