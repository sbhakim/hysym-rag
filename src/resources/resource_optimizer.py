# src/resources/resource_optimizer.py
import cvxpy as cp

class ResourceOptimizer:
    def optimize(self, cpu_usage, mem_usage, gpu_usage):
        """
        Optimize resource allocation using convex optimization.
        Args:
            cpu_usage (float): Current CPU usage as a fraction.
            mem_usage (float): Current memory usage (e.g., in GB or normalized value).
            gpu_usage (float): Current GPU usage as a fraction.
        Returns:
            dict: Optimal allocations for 'cpu' and 'gpu'.
        """
        # Decision variables: fractions for CPU and GPU allocations.
        cpu_alloc = cp.Variable()
        gpu_alloc = cp.Variable()

        # Constraints:
        constraints = [
            cpu_alloc + gpu_alloc <= 1.0,  # total allocation cannot exceed full capacity
            cpu_alloc >= 0.25,             # at least 25% CPU allocation required
            gpu_alloc <= 0.8,              # at most 80% GPU usage allowed
            cpu_alloc >= 0,
            gpu_alloc >= 0
        ]

        # Objective: minimize a weighted sum of resource allocations plus a term for memory usage.
        # (Memory usage is taken as a fixed input cost in this simple example.)
        objective = cp.Minimize(0.4 * cpu_alloc + 0.5 * gpu_alloc + 0.1 * mem_usage)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return {
            'cpu': cpu_alloc.value,
            'gpu': gpu_alloc.value
        }
