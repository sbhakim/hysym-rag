import logging
import torch
from typing import Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class UnifiedResponseAggregator:
    """
    Combines symbolic and neural answers into a single response with partial chain-of-thought.
    """

    def __init__(self, include_explanations: bool = True):
        self.include_explanations = include_explanations

    def aggregate(
            self,
            symbolic_responses: list,
            neural_response: str,
            chain_of_thought: Optional[list] = None
    ) -> str:
        """
        Produces a unified answer. Optionally includes brief chain-of-thought explanations.

        Args:
            symbolic_responses (list): List of extracted statements from symbolic reasoner.
            neural_response (str): The neural model's final answer string.
            chain_of_thought (list): Optional debug info on how symbolic reasoning progressed.

        Returns:
            str: A single combined response.
        """
        response_parts = []

        if symbolic_responses:
            response_parts.append("**Symbolic Reasoning Found:**")
            for i, resp in enumerate(symbolic_responses, start=1):
                response_parts.append(f"{i}. {resp}")

        response_parts.append("**Neural Model Suggests:**")
        response_parts.append(neural_response)

        if self.include_explanations and chain_of_thought:
            response_parts.append("\n**Chain of Thought (Symbolic Steps):**")
            for step in chain_of_thought:
                response_parts.append(f"- {step}")

        return "\n".join(response_parts)


class SystemControlManager:
    """
    Coordinates scheduling, fallback logic, and error handling for HySym-RAG.
    Works alongside the ResourceManager and HybridIntegrator.
    """

    def __init__(
            self,
            hybrid_integrator,
            resource_manager,
            aggregator: UnifiedResponseAggregator,
            error_retry_limit: int = 2,
            max_query_time: int = 10
    ):
        self.hybrid_integrator = hybrid_integrator
        self.resource_manager = resource_manager
        self.aggregator = aggregator
        self.error_retry_limit = error_retry_limit
        self.max_query_time = max_query_time

    def ensure_device_consistency(self, *tensors):
        """
        Ensure all tensors are moved to the same device as the primary component (neural model).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return [tensor.to(device) if tensor.device != device else tensor for tensor in tensors]

    def process_query_with_fallback(self, query: str, context: str) -> str:
        """
        Top-level method for handling a query with scheduling, fallback,
        error handling, and unified response generation.
        """
        start_time = datetime.now()
        attempts = 0
        symbolic_part = []
        neural_part = ""
        chain_of_thought = []
        final_response = "No answer generated."

        while attempts < self.error_retry_limit:
            attempts += 1
            try:
                # Check resource usage and adapt if needed
                usage = self.resource_manager.check_resources()
                self._adaptive_scheduling(usage)

                # Use the HybridIntegrator
                results, source = self.hybrid_integrator.process_query(query, context)

                # Ensure device consistency for tensors
                results = self.ensure_device_consistency(*results) if isinstance(results, torch.Tensor) else results

                # Gather results and aggregate
                if source in ("hybrid", "symbolic") and isinstance(results, list):
                    symbolic_part = results
                    chain_of_thought = results
                if source == "neural" and isinstance(results, list):
                    neural_part = results[0]
                if source == "hybrid" and symbolic_part and len(results) > 0:
                    neural_part = results[-1]

                final_response = self.aggregator.aggregate(
                    symbolic_responses=symbolic_part,
                    neural_response=neural_part,
                    chain_of_thought=chain_of_thought
                )
                break

            except RuntimeError as e:
                logger.error(f"Runtime error encountered: {str(e)} | Attempt: {attempts}")
                if "CUDA" in str(e):
                    logger.error("CUDA-related issue detected. Ensuring consistent device allocation.")
                    self.hybrid_integrator.ensure_device_consistency()
                continue

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)} | Attempt: {attempts}")
                continue

            time_elapsed = (datetime.now() - start_time).total_seconds()
            if time_elapsed > self.max_query_time or attempts >= self.error_retry_limit:
                final_response = (
                    f"Error encountered while processing. "
                    f"Attempts: {attempts}. Time Elapsed: {time_elapsed:.1f}s."
                )
                break

        return final_response

    def _adaptive_scheduling(self, usage: dict):
        """
        Example of scheduling logic based on usage.
        If GPU is heavily used, we might favor symbolic paths, etc.
        """
        gpu_usage = usage.get("gpu", 0.0)
        cpu_usage = usage.get("cpu", 0.0)

        if gpu_usage > self.resource_manager.current_thresholds['gpu']:
            logger.info("[SystemControlManager] High GPU usage detected, adjusting scheduling to symbolic preference.")
            self.hybrid_integrator.batch_size = 1
        elif cpu_usage > self.resource_manager.current_thresholds['cpu']:
            logger.info("[SystemControlManager] High CPU usage detected, reducing symbolic expansions.")
            if hasattr(self.hybrid_integrator.symbolic_reasoner, 'max_hops'):
                self.hybrid_integrator.symbolic_reasoner.max_hops = 2
        else:
            self.hybrid_integrator.batch_size = 4
            if hasattr(self.hybrid_integrator.symbolic_reasoner, 'max_hops'):
                self.hybrid_integrator.symbolic_reasoner.max_hops = 5
