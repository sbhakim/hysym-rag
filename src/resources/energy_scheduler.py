# In src/resources/energy_scheduler.py

import heapq


class EnergyAwareScheduler:
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.task_queue = []  # Priority queue for tasks

    def schedule_next_task(self):
        """Schedule the next task based on current resource usage.
        If no queued tasks, return a decision based on GPU usage.
        """
        if not self.task_queue:
            resources = self.resource_manager.check_resources()
            # For example, if GPU usage is high, fallback to symbolic reasoning
            if resources.get("gpu", 0) > 0.8:
                return "symbolic"
            return "neural"
        return self.task_queue.pop(0)

    def schedule_task(self, task_type, energy_threshold=0.8):
        """Legacy method for backward compatibility."""
        resources = self.resource_manager.check_resources()
        if task_type == "neural" and resources.get("gpu", 0) > energy_threshold:
            return "symbolic"
        return task_type
