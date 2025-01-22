import psutil

class ResourceManager:
    @staticmethod
    def check_resources():
        memory = psutil.virtual_memory()
        return {
            "available_memory": memory.available / (1024 ** 2),  # MB
            "used_memory": memory.used / (1024 ** 2)  # MB
        }
