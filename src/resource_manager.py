import psutil
import time

class ResourceManager:
    def __init__(self):
        """
        Initialize ResourceManager with default resource thresholds.
        """
        # Lowered CPU threshold slightly for demonstration
        self.resource_thresholds = {
            'memory_threshold': 0.8,  # 80% memory utilization
            'cpu_threshold': 0.75    # 75% CPU utilization (was 0.9)
        }
        print("ResourceManager initialized with default thresholds.")

    @staticmethod
    def check_resources():
        """
        Check the current system resource usage.
        """
        memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=1)
        return {
            "available_memory_mb": memory.available / (1024 ** 2),
            "used_memory_mb": memory.used / (1024 ** 2),
            "memory_utilization": memory.percent / 100.0,
            "cpu_usage": cpu_usage / 100.0
        }

    def should_use_symbolic(self, query_complexity):
        """
        Decide whether to use symbolic reasoning based on resource usage and query complexity.
        """
        resources = self.check_resources()

        # If system usage is high, force symbolic
        if (resources['memory_utilization'] > self.resource_thresholds['memory_threshold'] or
                resources['cpu_usage'] > self.resource_thresholds['cpu_threshold']):
            print("High resource usage detected. Using symbolic reasoning.")
            return True

        # For demonstration, if the query complexity is below 1.0, prefer symbolic
        if query_complexity < 2.0:
            print("Query complexity is moderate or low. Using symbolic reasoning.")
            return True

        # Otherwise, default to neural
        print("Defaulting to neural reasoning.")
        return False

    @staticmethod
    def monitor_resource_usage(during_inference):
        """
        Monitor memory and time usage during inference.
        """
        start_memory = psutil.virtual_memory().used
        start_time = time.time()

        during_inference()

        end_memory = psutil.virtual_memory().used
        end_time = time.time()

        return {
            "memory_used_mb": round((end_memory - start_memory) / (1024 ** 2), 4),
            "time_taken_s": round(end_time - start_time, 4),
        }
