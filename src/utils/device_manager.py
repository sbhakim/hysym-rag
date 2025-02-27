# src/utils/device_manager.py

import torch

class DeviceManager:
    @staticmethod
    def get_device():
        """Return a consistent device (cuda if available, else cpu)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def ensure_same_device(tensor1, tensor2, device=None):
        """
        Move both tensors to the same device.
        If no device is specified, use the device of the first tensor.
        """
        if device is None:
            device = tensor1.device
        return tensor1.to(device), tensor2.to(device)
