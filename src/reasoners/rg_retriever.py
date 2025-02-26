# src/reasoners/rg_retriever.py

from sentence_transformers import SentenceTransformer, util
import torch
import logging
from typing import List, Dict, Optional, Any, Tuple
from src.utils.device_manager import DeviceManager  # Import DeviceManager
import re

logger = logging.getLogger(__name__)


class RuleGuidedRetriever:
    def __init__(self, encoder=None, similarity_threshold=0.3, adaptive_threshold=True, context_granularity="sentence"):
        """
        Advanced Rule-Guided Retriever using semantic similarity for context filtering.

        Args:
            encoder: SentenceTransformer model for embedding sentences and rules.
            similarity_threshold: Base threshold for semantic similarity.
            adaptive_threshold: Whether to use adaptive similarity threshold based on query complexity.
            context_granularity: "sentence" or "paragraph" - level of context filtering.
        """
        self.encoder = encoder if encoder is not None else SentenceTransformer('all-MiniLM-L6-v2')
        self.base_similarity_threshold = similarity_threshold
        self.adaptive_threshold = adaptive_threshold
        self.context_granularity = context_granularity
        self.device = DeviceManager.get_device()
        self.encoder.to(self.device)

    def filter_context_by_rules(self, context: str, symbolic_guidance: List[Dict[str, Any]],
                                query_complexity: float = 0.5,
                                supporting_facts: Optional[List[Tuple[str, int]]] = None) -> str:
        """
        Enhanced filter_context_by_rules with confidence-weighted filtering and supporting fact prioritization.
        """
        if not symbolic_guidance:
            logger.info("Advanced RG-Retriever: No rules provided, skipping filtering.")
            return context

        # Process rules and gather confidence scores
        rule_embeddings_with_confidence = []

        # Get high-confidence segments first (for supporting facts)
        prioritized_segments = []
        if supporting_facts:
            context_segments = self._split_context(context)  # Use helper function to split context
            # Map supporting facts to segments
            for fact_text, _ in supporting_facts:
                best_segment = None
                best_score = 0
                for segment in context_segments:
                    similarity = self._calculate_text_similarity(segment,
                                                                 fact_text)  # Use helper function for similarity
                    if similarity > best_score:
                        best_score = similarity
                        best_segment = segment
                if best_segment and best_score > 0.5:  # Reasonable threshold
                    prioritized_segments.append(best_segment)

        # Process rules and extract confidence scores
        for rule in symbolic_guidance:
            # Handle string rules
            if isinstance(rule, str):
                rule_text = rule.strip()
                if not rule_text:
                    continue
                rule_confidence = 0.7  # Default confidence for string rules
                try:
                    rule_embedding = self.encoder.encode(rule_text, convert_to_tensor=True).to(self.device)
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
                    rule_embedding = self.encoder.encode(rule_text, convert_to_tensor=True).to(self.device)
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

        # Calculate total confidence for normalization
        confidence_sum = sum(rule_data['confidence'] ** 2 for rule_data in rule_embeddings_with_confidence)
        if confidence_sum == 0:
            confidence_sum = 1.0  # Avoid division by zero

        # Adaptive Threshold Calculation with lower base for better recall
        similarity_threshold = self.base_similarity_threshold
        if self.adaptive_threshold:
            # Lower the base threshold to capture more context
            similarity_threshold = max(0.25, self.base_similarity_threshold * (1 - query_complexity * 0.5))
        logger.debug(f"Advanced RG-Retriever: Using similarity threshold: {similarity_threshold:.2f}")

        # Split context based on granularity
        context_segments = self._split_context(context)  # Use helper function

        # Track best segments with their scores
        scored_segments = []

        # Process each segment and score it
        for segment in context_segments:
            try:
                # Skip very short segments
                if len(segment.strip()) < 10:
                    continue

                # Use encode with explicit device management
                segment_embedding = self.encoder.encode(segment, convert_to_tensor=True).to(self.device)

                # Calculate confidence-weighted similarity score
                weighted_score = 0.0
                for rule_data in rule_embeddings_with_confidence:
                    rule_emb = rule_data['embedding']
                    rule_confidence = rule_data['confidence']
                    # Calculate cosine similarity
                    similarity = util.cos_sim(segment_embedding, rule_emb).item()

                    # Apply confidence as weight (quadratic scaling)
                    weighted_score += similarity * (rule_confidence ** 2 / confidence_sum)  # Quadratic scaling

                scored_segments.append((weighted_score, segment))

            except Exception as e:
                logger.error(f"Error encoding segment '{segment[:50]}...': {e}")
                continue

        # Sort segments by similarity score
        scored_segments.sort(reverse=True, key=lambda x: x[0])

        # Always include at least the top 3 segments, regardless of threshold
        filtered_context_segments = prioritized_segments.copy()  # Start with prioritized segments

        # Add top-scored segments (avoid duplicates)
        for score, segment in scored_segments[:min(3, len(scored_segments))]:
            if segment not in filtered_context_segments:
                filtered_context_segments.append(segment)

        # Add more segments that meet the threshold (avoid duplicates)
        for score, segment in scored_segments[3:]:
            if score >= similarity_threshold and segment not in filtered_context_segments:
                filtered_context_segments.append(segment)

        # Fallback: if still no segments, use original context segments
        if not filtered_context_segments and context_segments:
            logger.info("Advanced RG-Retriever: No matching segments found. Using top 3 context segments as fallback.")
            filtered_context_segments = context_segments[:min(3, len(context_segments))]

        # Final Fallback: If even the best segment isn't found, return the original context.
        if not filtered_context_segments:
            logger.warning("Advanced RG-Retriever: No segments found, returning original context.")
            return context

        # Reconstruct filtered context
        if self.context_granularity == "paragraph":
            filtered_context = "\n\n".join(filtered_context_segments)
        else:
            filtered_context = ". ".join(filtered_context_segments)

        # Calculate reduction percentage for analytics
        original_len = len(context)
        filtered_len = len(filtered_context)
        reduction = 1 - (filtered_len / original_len) if original_len > 0 else 0

        logger.info(
            f"Advanced RG-Retriever ({self.context_granularity.capitalize()}, Semantic): Context filtered "
            f"(threshold={similarity_threshold:.2f}, adaptive={self.adaptive_threshold}). "
            f"Original length: {original_len}, Filtered length: {filtered_len}, Reduction: {reduction * 100:.1f}%")

        return filtered_context

    def _split_context(self, context: str) -> List[str]:  # Helper function to split context
        """Split context into segments based on granularity setting."""
        if self.context_granularity == "paragraph":
            return [seg.strip() for seg in context.split('\n\n') if seg.strip()]
        else:  # sentence level split
            # More robust sentence splitting
            return [seg.strip() for seg in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context) if
                    seg.strip()]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:  # Helper function for similarity calculation
        """Calculate semantic similarity between two text segments."""
        try:
            emb1 = self.encoder.encode(text1, convert_to_tensor=True).to(self.device)
            emb2 = self.encoder.encode(text2, convert_to_tensor=True).to(self.device)
            return util.cos_sim(emb1, emb2).item()
        except Exception as e:
            logger.warning(f"Error calculating text similarity: {e}")
            # Fallback to basic overlap measure
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            overlap = len(words1.intersection(words2))
            return overlap / max(len(words1), len(words2))


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