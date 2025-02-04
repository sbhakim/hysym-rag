# src/power_monitor.py
from collections import defaultdict
import time

class PowerMonitor:
    def __init__(self):
        self.energy_stats = defaultdict(float)
        self.start_times = {}
        self.component_specs = {
            'gpu': {'idle_watts': 30, 'load_watts': 250},
            'cpu': {'idle_watts': 10, 'load_watts': 65},
            'symbolic': {'idle_watts': 1, 'load_watts': 5}  # Estimated for symbolic reasoning
        }

    def start_tracking(self, component):
        """Start tracking energy usage for a component."""
        self.start_times[component] = time.time()

    def stop_tracking(self, component, utilization):
        """Stop tracking energy usage and calculate the energy consumed."""
        duration = time.time() - self.start_times[component]
        specs = self.component_specs[component]

        # Calculate power consumption based on load and idle wattage
        power_draw = (specs['load_watts'] * utilization + specs['idle_watts'] * (1 - utilization))
        energy_consumed = power_draw * duration / 3600  # Convert to watt-hours

        self.energy_stats[component] += energy_consumed
        return energy_consumed

    def get_energy_stats(self):
        """Return the energy stats collected."""
        return dict(self.energy_stats)
