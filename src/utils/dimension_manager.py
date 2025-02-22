# src/utils/dimension_manager.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DimensionalityManager:
    """
    Manages embedding dimensions across the system to ensure consistency.
    Provides caching and validation of dimension transformations.
    """

    def __init__(self, target_dim: int = 768, device: Optional[torch.device] = None):
        self.target_dim = target_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alignment_cache: Dict[str, nn.Module] = {}
        self.mismatch_counts: Dict[str, int] = defaultdict(int)

    def register_adapter(self, name: str, adapter: nn.Module) -> None:
        """
        Register a pre-initialized adapter module for a given source.
        """
        self.alignment_cache[name] = adapter.to(self.device)
        logger.info(f"Adapter registered for {name} with target_dim {self.target_dim}")

    def align_embeddings(self,
                         embedding: torch.Tensor,
                         source: str,
                         return_confidence: bool = False) -> torch.Tensor:
        """
        Aligns embeddings to target dimension using registered adapters
        or creating new ones as needed.

        Args:
            embedding: Input embedding tensor.
            source: Source identifier (e.g., 'symbolic', 'neural').
            return_confidence: Whether to return an alignment confidence score.

        Returns:
            Aligned embedding tensor, optionally with a confidence score.
        """
        # Ensure embedding is at least 2D.
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        input_dim = embedding.size(-1)

        # If already aligned, return immediately.
        if input_dim == self.target_dim:
            return (embedding, 1.0) if return_confidence else embedding

        # Use registered adapter if available.
        if source in self.alignment_cache:
            projection = self.alignment_cache[source]
        else:
            # Create and register new adapter.
            projection = self._create_projection(input_dim)
            self.register_adapter(source, projection)
            self.mismatch_counts[f"{input_dim}_{source}"] += 1

        try:
            aligned = projection(embedding)
            confidence = self._calculate_alignment_confidence(embedding, aligned)
            return (aligned, confidence) if return_confidence else aligned
        except Exception as e:
            logger.error(f"Alignment failed for source {source}: {str(e)}")
            raise

    def _create_projection(self, input_dim: int) -> nn.Module:
        """
        Creates a new projection layer to map embeddings from input_dim to target_dim.
        """
        projection = nn.Sequential(
            nn.Linear(input_dim, self.target_dim),
            nn.LayerNorm(self.target_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        return projection.to(self.device)

    def _calculate_alignment_confidence(self,
                                        original: torch.Tensor,
                                        aligned: torch.Tensor) -> float:
        """
        Calculates a confidence score for the alignment based on cosine similarity.
        """
        with torch.no_grad():
            if original.size(-1) != aligned.size(-1):
                # Create a temporary projector if dimensions differ.
                projector = nn.Linear(original.size(-1), aligned.size(-1)).to(original.device)
                original = projector(original)
            similarity = F.cosine_similarity(original, aligned).mean().item()
            return max(0.0, min(1.0, similarity))

    def get_alignment_stats(self) -> Dict:
        """
        Returns statistics about registered adapters and dimension mismatches.
        """
        return {
            "total_adapters": len(self.alignment_cache),
            "mismatch_counts": dict(self.mismatch_counts)
        }