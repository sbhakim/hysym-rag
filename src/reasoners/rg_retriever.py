# src/reasoners/rg_retriever.py

from sentence_transformers import SentenceTransformer, util
import torch
import logging
from typing import List, Dict, Optional, Any, Tuple
from src.utils.device_manager import DeviceManager  # Import DeviceManager

logger = logging.getLogger(__name__)


class RuleGuidedRetriever:
    def __init__(self, encoder=None, similarity_threshold=0.3, adaptive_threshold=True, context_granularity="sentence"):
        """
        Advanced Rule-Guided Retriever using semantic similarity for context filtering.

        Args:
            encoder: SentenceTransformer model for embedding sentences and rules.
            similarity_threshold: Base threshold for semantic similarity.
            adaptive_threshold: Whether to use adaptive similarity threshold based on query complexity (basic example).
            context_granularity: "sentence" or "paragraph" - level of context filtering.
        """
        self.encoder = encoder if encoder is not None else SentenceTransformer('all-MiniLM-L6-v2')
        self.base_similarity_threshold = similarity_threshold  # Renamed to base for adaptive use
        self.adaptive_threshold = adaptive_threshold  # Flag for adaptive thresholding
        self.context_granularity = context_granularity  # "sentence" or "paragraph"
        self.device = DeviceManager.get_device()  # Use DeviceManager
        self.encoder.to(self.device)  # Move encoder to the managed device

    def filter_context_by_rules(self, context: str, symbolic_guidance: List[Dict[str, Any]],
                                query_complexity: float = 0.5,
                                supporting_facts: Optional[List[Tuple[str, int]]] = None) -> str:
        """
        Filters the context based on semantic similarity to symbolic rules.

        Args:
            context: The original context text (string).
            symbolic_guidance: List of symbolic rules (dicts) for guidance. Each dict should have a "response" key.
            query_complexity: Score representing the complexity of the query (used for adaptive threshold).
            supporting_facts:  Optional list of supporting facts, not directly used here, but passed through.

        Returns:
            string: Filtered context text.  Returns the original context if no filtering occurs or all filtering fails.
        """
        if not symbolic_guidance:
            logger.info("Advanced RG-Retriever: No rules provided, skipping filtering.")
            return context

        # More flexible rule format handling
        rule_embeddings_with_confidence = []
        for rule in symbolic_guidance:
            # Handle string rules
            if isinstance(rule, str):
                rule_text = rule.strip()
                if not rule_text:
                    continue
                rule_confidence = 0.7  # Default confidence for string rules
                try:
                    rule_embedding = self.encoder.encode(rule_text, convert_to_tensor=True, device=self.device)
                    rule_embeddings_with_confidence.append({'embedding': rule_embedding, 'confidence': rule_confidence})
                except Exception as e:
                    logger.debug(f"Error encoding rule: {e}")
                    continue
            # Handle dictionary rules
            elif isinstance(rule, dict):
                # Look for response content in multiple possible fields
                rule_text = None
                for field in ['response', 'statement', 'text', 'source_text', 'content']:
                    if field in rule and rule[field]:
                        rule_text = rule[field]
                        break

                if not rule_text:
                    continue

                rule_confidence = rule.get('confidence', 0.7)
                try:
                    rule_embedding = self.encoder.encode(rule_text, convert_to_tensor=True, device=self.device)
                    rule_embeddings_with_confidence.append({'embedding': rule_embedding, 'confidence': rule_confidence})
                except Exception as e:
                    logger.debug(f"Error encoding rule: {e}")
                    continue
            else:
                logger.warning(f"Unsupported rule format: {type(rule)}. Skipping.")
                continue

        if not rule_embeddings_with_confidence:
            logger.info("Advanced RG-Retriever: No valid rule embeddings, skipping filtering.")
            return context

        # Adaptive Threshold Calculation
        similarity_threshold = self.base_similarity_threshold
        if self.adaptive_threshold:
            similarity_threshold = self.base_similarity_threshold + (0.6 - self.base_similarity_threshold) * (
                        1 - query_complexity)
        logger.debug(f"Advanced RG-Retriever: Using similarity threshold: {similarity_threshold:.2f}")

        # Split context based on granularity
        if self.context_granularity == "paragraph":
            context_segments = [seg.strip() for seg in context.split('\n\n') if seg.strip()]
        else:
            context_segments = [seg.strip() for seg in context.split('.') if seg.strip()]

        filtered_context_segments = []

        # Process each segment
        for segment in context_segments:
            try:
                # Use encode with explicit device management
                segment_embedding = self.encoder.encode(segment, convert_to_tensor=True, device=self.device)
            except Exception as e:
                logger.error(f"Error encoding segment '{segment[:50]}...': {e}")  # Log first 50 chars
                continue

            weighted_similarity_sum = 0.0
            for rule_data in rule_embeddings_with_confidence:
                rule_emb = rule_data['embedding']
                rule_confidence = rule_data['confidence']
                # No need for ensure_same_device here; embeddings created on the correct device above.
                similarity = util.cos_sim(segment_embedding, rule_emb).item()
                weighted_similarity_sum += similarity * rule_confidence

            # Apply filtering
            if weighted_similarity_sum >= similarity_threshold * len(rule_embeddings_with_confidence):
                filtered_context_segments.append(segment)

        # Fallback: if no segments passed, use the best matching segment.
        if not filtered_context_segments and context_segments:
            best_segment = None
            best_score = -1.0
            for segment in context_segments:
                try:
                    # Use encode with explicit device management
                    segment_embedding = self.encoder.encode(segment, convert_to_tensor=True, device=self.device)
                except Exception as e:
                    logger.error(f"Error encoding segment (fallback) '{segment[:50]}...': {e}")
                    continue

                weighted_similarity_sum = 0.0
                for rule_data in rule_embeddings_with_confidence:
                    rule_emb = rule_data['embedding']
                    rule_confidence = rule_data['confidence']
                    # No need for ensure_same_device here.
                    similarity = util.cos_sim(segment_embedding, rule_emb).item()
                    weighted_similarity_sum += similarity * rule_confidence

                if weighted_similarity_sum > best_score:
                    best_score = weighted_similarity_sum
                    best_segment = segment

            if best_segment:
                filtered_context_segments.append(best_segment)
                logger.info(
                    "Advanced RG-Retriever: No segments met the threshold; using best matching segment as fallback.")

        # Final Fallback: If even the best segment isn't found, return the original context.
        if not filtered_context_segments:
            logger.warning("Advanced RG-Retriever: No segments found, returning original context.")
            return context

        # Reconstruct filtered context
        if self.context_granularity == "paragraph":
            filtered_context = "\n\n".join(filtered_context_segments)
        else:
            filtered_context = ". ".join(filtered_context_segments)

        logger.info(
            f"Advanced RG-Retriever ({self.context_granularity.capitalize()}, Semantic): Context filtered (threshold={similarity_threshold:.2f}, adaptive={self.adaptive_threshold}). Original length: {len(context)}, Filtered length: {len(filtered_context)}")
        return filtered_context


if __name__ == '__main__':
    # Example Usage and Testing (Optional)
    retriever = RuleGuidedRetriever(similarity_threshold=0.5, adaptive_threshold=True, context_granularity="paragraph")
    sample_context = """
    Deforestation has significant environmental impacts. Soil erosion increases after deforestation.

    Biodiversity loss is a major concern due to habitat destruction. Climate change is accelerated by deforestation.

    Sustainable forestry is crucial. Reforestation can help mitigate deforestation effects.
    """
    sample_rules = [
        {"response": "Deforestation leads to soil erosion.", "confidence": 0.95},
        {"response": "Biodiversity is reduced by deforestation.", "confidence": 0.85},
        {"response": "Climate change is linked to deforestation."}  # Now consistent format
    ]
    sample_query_complexity = 0.7

    filtered_context = retriever.filter_context_by_rules(sample_context, sample_rules, sample_query_complexity)
    print("\nOriginal Context:\n", sample_context)
    print("\nFiltered Context:\n", filtered_context)