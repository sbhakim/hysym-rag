# src/integrators/hybrid_integrator.py
import time
import logging
import torch
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
from sentence_transformers import util

from src.knowledge_integrator import AlignmentLayer
#from src.utils.rule_extractor import extract_rules_from_neural_output
from src.reasoners.rg_retriever import RuleGuidedRetriever  # Import Advanced RG-Retriever

logger = logging.getLogger(__name__)


class HybridIntegrator:
    """
    HySym-RAG's core integration system that combines symbolic and neural reasoning.
    """

    def __init__(self, symbolic_reasoner, neural_retriever, resource_manager, query_expander=None, cache_ttl=3600,
                 batch_size=4):
        self.symbolic_reasoner = symbolic_reasoner
        self.neural = neural_retriever
        self.resource_manager = resource_manager
        self.query_expander = query_expander
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        self.performance_stats = {
            'symbolic_hits': 0,
            'neural_hits': 0,
            'cache_hits': 0,
            'total_queries': 0,
            'hybrid_successes': 0
        }
        self.alignment_layer = AlignmentLayer(sym_dim=300, neural_dim=384, target_dim=768)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("HybridIntegrator initialized successfully with alignment layer")
        # Initialize Advanced RG-Retriever (if needed)
        self.advanced_rg_retriever = RuleGuidedRetriever()

    @property
    def expander(self):
        return self.query_expander

    def ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def ensure_device_consistency(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        return [self.ensure_device(tensor) for tensor in tensors]

    def _generate_cache_key(self, query: str) -> str:
        if self.query_expander:
            expanded = self.query_expander.expand_query(query)
            return f"{query}::{expanded}"
        return query

    def _get_valid_cache(self, cache_key: str) -> Optional[Tuple[List[str], str]]:
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self.cache_ttl):
                self.performance_stats['cache_hits'] += 1
                return entry['result'], entry['source']
            else:
                del self.cache[cache_key]
        return None

    def _update_cache(self, cache_key: str, result: Tuple[List[str], str]) -> None:
        if len(self.cache) > 1000:
            oldest_key = min(self.cache.items(), key=lambda x: x[1]['timestamp'])[0]
            del self.cache[oldest_key]
        self.cache[cache_key] = {
            'result': result[0],
            'source': result[1],
            'timestamp': datetime.now()
        }

    def process_query(self, query: str, context: str, query_complexity=0.5) -> Tuple[str, str]:
        """
        Process a query using hybrid reasoning.

        Returns:
            Tuple[str, str]: (response text, reasoning source)
        """
        try:
            # Get symbolic result
            symbolic_result = self._process_symbolic(query)

            # Simple queries use symbolic path only
            if query_complexity < 0.4 and symbolic_result:
                response = " ".join(symbolic_result) if isinstance(symbolic_result, list) else symbolic_result
                return response, "symbolic"

            # For more complex queries, try hybrid approach
            if symbolic_result:
                neural_response = self.neural.retrieve_answer(
                    context,
                    query,
                    symbolic_guidance=symbolic_result,
                    query_complexity=query_complexity
                )
                # Combine responses
                final_response = f"{' '.join(symbolic_result) if isinstance(symbolic_result, list) else symbolic_result}\n{neural_response}"
                return final_response, "hybrid"

            # Fallback to pure neural
            neural_response = self.neural.retrieve_answer(context, query)
            return neural_response, "neural"

        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return "Error processing query.", "error"

    def _prepare_embedding(self, embedding: torch.Tensor, target_dim: int, name: str) -> torch.Tensor:
        """Prepare embedding (unchanged)."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.size(1) != target_dim:
            logger.warning(f"Transforming {name} embedding from {embedding.size(1)} to {target_dim}")
            projection = torch.nn.Linear(embedding.size(1), target_dim).to(self.device)
            embedding = projection(embedding)
        return embedding

    def _combine_results(self, symbolic_results: List[str], neural_result: List[str]) -> List[str]:
        """Combine symbolic and neural results."""
        combined = []
        combined.extend(symbolic_results)
        for neural_resp in neural_result:
            if not any(self._high_overlap(neural_resp, sym_resp) for sym_resp in symbolic_results):
                combined.append(neural_resp)
        return combined

    def _high_overlap(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check for high overlap between two texts using cosine similarity."""
        emb1 = self.neural.encode(text1)
        emb2 = self.neural.encode(text2)
        similarity = util.cos_sim(emb1, emb2).item()
        return similarity > threshold

    def _process_symbolic(self, query: str) -> Optional[List[str]]:
        """Process the query symbolically."""
        try:
            symbolic_results = self.symbolic_reasoner.process_query(query)
            if symbolic_results and "No symbolic match found." not in symbolic_results:
                self.performance_stats['symbolic_hits'] += 1
                return symbolic_results
            return None
        except Exception as e:
            logger.error(f"Error in symbolic processing: {str(e)}")
            return None

    def _process_neural_optimized(self, query: str, context: str, symbolic_guidance: Optional[List[str]] = None,
                                  query_complexity=0.5) -> List[str]:
        """Process the query neurally using advanced RG-Retriever."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            start_time = time.time()
            # Use Advanced RG-Retriever for context filtering, passing query_complexity
            filtered_context = self.neural.advanced_rg_retriever.filter_context_by_rules(
                context, symbolic_guidance, query_complexity=query_complexity
            )
            # Retrieve neural answer and ensure the return is a 2-tuple
            result = self.neural.retrieve_answer(
                filtered_context,
                query,
                symbolic_guidance=symbolic_guidance,
                rule_guided_retrieval=False,
                query_complexity=query_complexity
            )
            if isinstance(result, tuple):
                if len(result) != 2:
                    logger.error(f"Unexpected tuple length from retrieve_answer: {result}")
                    neural_answer = result[0]
                else:
                    neural_answer, neural_metadata = result
            else:
                neural_answer = result

            result_list = [neural_answer]
            dynamic_rules = extract_rules_from_neural_output(neural_answer)
            if dynamic_rules:
                logger.info(f"Extracted {len(dynamic_rules)} dynamic rules from neural output.")
                self.symbolic_reasoner.add_dynamic_rules(dynamic_rules)
            processing_time = time.time() - start_time
            self.resource_manager.neural_perf_times.append(processing_time)
            self.performance_stats['neural_hits'] += 1
            return result_list
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Return performance metrics."""
        total = max(1, self.performance_stats['total_queries'])
        return {
            'symbolic_ratio': self.performance_stats['symbolic_hits'] / total,
            'neural_ratio': self.performance_stats['neural_hits'] / total,
            'cache_hit_ratio': self.performance_stats['cache_hits'] / total,
            'hybrid_success_ratio': self.performance_stats['hybrid_successes'] / total,
            'average_neural_time': (
                sum(self.resource_manager.neural_perf_times) / len(self.resource_manager.neural_perf_times)
                if self.resource_manager.neural_perf_times else 0
            )
        }

    def _handle_runtime_error(self, error: RuntimeError):
        """Handle runtime errors."""
        logger.error(f"Runtime error occurred: {str(error)}")
        # Additional handling if needed

    def _handle_fallback(self, query: str, context: str) -> Tuple[List[str], str]:
        """Return a fallback response in case of failure."""
        fallback_response = ["Unable to process query. Please try again later."]
        logger.info(f"_handle_fallback returning: {(fallback_response, 'fallback')}")
        return fallback_response, "fallback"
