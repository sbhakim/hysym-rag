# src/config/config_manager.py

import yaml
import psutil
import torch

class ConfigManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set memory limits based on system specs
        total_memory = psutil.virtual_memory().total / (1024**3)  # in GB
        if torch.cuda.is_available():
            gpu_properties = torch.cuda.get_device_properties(0)
            gpu_memory = gpu_properties.total_memory / (1024**3)
        else:
            gpu_memory = 0

        self.config['memory_limits'] = {
            'ram_limit': min(int(total_memory * 0.8), 256),  # Use up to 256GB or 80%
            'gpu_limit': min(int(gpu_memory * 0.8), 20) if gpu_memory else 0,  # Use up to 20GB or 80%
            'batch_size': self._calculate_optimal_batch_size(gpu_memory)
        }

    def _calculate_optimal_batch_size(self, gpu_memory):
        """Determine optimal batch size based on available GPU memory."""
        base_memory_per_sample = 0.5  # GB per sample (estimate)
        if gpu_memory:
            return max(1, int(gpu_memory * 0.3 / base_memory_per_sample))
        return 1

    def get_config(self):
        return self.config

