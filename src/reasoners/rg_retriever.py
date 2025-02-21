# src/reasoners/rg_retriever.py

from sentence_transformers import SentenceTransformer, util
import torch
import logging

logger = logging.getLogger(__name__)


class RuleGuidedRetriever:
    def __init__(self, encoder=None, similarity_threshold=0.4, adaptive_threshold=True, context_granularity="sentence"):
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

    def filter_context_by_rules(self, context, symbolic_guidance, query_complexity=0.5):
        """
        Filters the context based on semantic similarity to symbolic rules.

        Args:
            context: The original context text (string).
            symbolic_guidance: List of symbolic rules (dicts or strings) for guidance.
            query_complexity: Score representing the complexity of the query (used for adaptive threshold).

        Returns:
            string: Filtered context text.
        """
        if not symbolic_guidance:
            logger.info("Advanced RG-Retriever: No rules provided, skipping filtering.")
            return context

        rule_embeddings_with_confidence = []  # Store embeddings and confidence
        for rule in symbolic_guidance:
            rule_text = ""
            rule_confidence = 1.0  # Default confidence
            if isinstance(rule, dict):
                rule_text = rule.get("response", str(rule))
                rule_confidence = rule.get("confidence", 1.0)  # Get rule confidence if available
            elif isinstance(rule, str):
                rule_text = rule
            else:
                rule_text = ""
            if rule_text:
                rule_embedding = self.encoder.encode(rule_text, convert_to_tensor=True)
                rule_embeddings_with_confidence.append({'embedding': rule_embedding, 'confidence': rule_confidence})

        if not rule_embeddings_with_confidence:
            logger.info("Advanced RG-Retriever: No rule embeddings available, skipping filtering.")
            return context

        # Adaptive Threshold Calculation (Basic example based on query_complexity)
        similarity_threshold = self.base_similarity_threshold
        if self.adaptive_threshold:
            similarity_threshold = self.base_similarity_threshold + (0.6 - self.base_similarity_threshold) * (
                        1 - query_complexity)

        filtered_context_segments = []  # Segments can be sentences or paragraphs

        # Split context based on granularity
        if self.context_granularity == "paragraph":
            context_segments = [seg.strip() for seg in context.split('\n\n') if seg.strip()]
        else:
            context_segments = [seg.strip() for seg in context.split('.') if seg.strip()]

        # Process each segment: calculate weighted similarity across all rules
        for segment in context_segments:
            segment_embedding = self.encoder.encode(segment, convert_to_tensor=True)
            weighted_similarity_sum = 0.0  # For ranked rule usage
            for rule_data in rule_embeddings_with_confidence:
                rule_emb = rule_data['embedding']
                rule_confidence = rule_data['confidence']
                similarity = util.cos_sim(segment_embedding, rule_emb).item()
                weighted_similarity_sum += similarity * rule_confidence
            # Using weighted similarity sum and adaptive threshold for segment filtering.
            if weighted_similarity_sum >= similarity_threshold * len(rule_embeddings_with_confidence):
                filtered_context_segments.append(segment)

        # Fallback: if no segments passed the filter, include the segment with the highest weighted similarity.
        if not filtered_context_segments and context_segments:
            best_segment = None
            best_score = -1.0
            for segment in context_segments:
                segment_embedding = self.encoder.encode(segment, convert_to_tensor=True)
                weighted_similarity_sum = 0.0
                for rule_data in rule_embeddings_with_confidence:
                    rule_emb = rule_data['embedding']
                    rule_confidence = rule_data['confidence']
                    similarity = util.cos_sim(segment_embedding, rule_emb).item()
                    weighted_similarity_sum += similarity * rule_confidence
                if weighted_similarity_sum > best_score:
                    best_score = weighted_similarity_sum
                    best_segment = segment
            if best_segment:
                filtered_context_segments.append(best_segment)
                logger.info(
                    "Advanced RG-Retriever: No segments met the threshold; using best matching segment as fallback.")

        # Reconstruct filtered context based on granularity
        if filtered_context_segments:
            if self.context_granularity == "paragraph":
                filtered_context = "\n\n".join(filtered_context_segments)
            else:
                filtered_context = ". ".join(filtered_context_segments)
            logger.info(
                f"Advanced RG-Retriever ({self.context_granularity.capitalize()}, Semantic): Context filtered (threshold={similarity_threshold:.2f}, adaptive={self.adaptive_threshold}). Original length: {len(context)}, Filtered length: {len(filtered_context)}")
            return filtered_context
        else:
            logger.info(
                f"Advanced RG-Retriever ({self.context_granularity.capitalize()}, Semantic): No context filtering applied - no segments above similarity threshold.")
            return ""


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
        "Climate change is linked to deforestation."
    ]
    sample_query_complexity = 0.7

    filtered_context = retriever.filter_context_by_rules(sample_context, sample_rules, sample_query_complexity)
    print("\nOriginal Context:\n", sample_context)
    print("\nFiltered Context:\n", filtered_context)
