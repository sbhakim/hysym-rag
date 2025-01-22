import psutil
import time

class ResourceManager:
    @staticmethod
    def check_resources():
        memory = psutil.virtual_memory()
        return {
            "available_memory_mb": memory.available / (1024 ** 2),
            "used_memory_mb": memory.used / (1024 ** 2),
        }

    @staticmethod
    def monitor_resource_usage(during_inference):
        start_memory = psutil.virtual_memory().used
        start_time = time.time()

        during_inference()

        end_memory = psutil.virtual_memory().used
        end_time = time.time()

        return {
            "memory_used_mb": (end_memory - start_memory) / (1024 ** 2),
            "time_taken_s": end_time - start_time,
        }
