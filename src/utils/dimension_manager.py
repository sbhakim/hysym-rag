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
        # Add a validation layer for explicit 384 to 768 conversion
        self.validation_layer = nn.Sequential(
            nn.Linear(384, target_dim),  # Explicit conversion from 384 to 768
            nn.LayerNorm(target_dim),
            nn.ReLU()
        ).to(self.device)
        logger.info(f"DimensionalityManager initialized: target_dim={target_dim}, device={self.device}")

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
        Enhanced embedding alignment with improved validation and error handling.
        """
        try:
            # Ensure embedding is at least 2D.
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            input_dim = embedding.size(-1)
            logger.debug(
                f"Aligning embedding from source '{source}', original dim={input_dim}, target_dim={self.target_dim}, current embedding shape: {embedding.shape}")

            # If already aligned, return immediately.
            if input_dim == self.target_dim:
                logger.debug(f"Embedding from source '{source}' already aligned (dim={input_dim}).")
                return (embedding, 1.0) if return_confidence else embedding

            # Use registered adapter if available.
            if source in self.alignment_cache:
                adapter = self.alignment_cache[source]
                logger.debug(f"Using registered adapter for source '{source}'.")
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # Ensure 2D input for adapter
                aligned = adapter(embedding)
                if aligned.shape[-1] != self.target_dim:
                    raise ValueError(
                        f"Aligned embedding dimension {aligned.shape[-1]} from source '{source}' does not match target {self.target_dim}")
                logger.debug(f"Aligned embedding shape using adapter: {aligned.shape}")
                return aligned

            # Special handling for symbolic embeddings.
            if input_dim == 384:
                logger.debug(
                    f"Aligning symbolic embedding from source '{source}' (dim=384) to target dim {self.target_dim} using validation_layer.")
                aligned = self.validation_layer(embedding)
                logger.debug(f"Aligned embedding shape using validation_layer: {aligned.shape}")
                return aligned

            # Dynamic projection for unknown dimensions.
            logger.debug(
                f"Aligning embedding from source '{source}' (dim={input_dim}) to target dim {self.target_dim} using dynamic projection.")
            projection = self._create_projection(input_dim)
            projected_embedding = projection(embedding)
            logger.debug(f"Aligned embedding shape using dynamic projection: {projected_embedding.shape}")
            return projected_embedding

        except Exception as e:
            logger.error(f"Error in alignment: {str(e)}")
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
                # Create a temporary projector if dimensions differ (should not happen if correctly configured).
                projector = nn.Linear(original.size(-1), aligned.size(-1)).to(original.device)
                original = projector(original)
                logger.warning("Dimension mismatch in _calculate_alignment_confidence, using temporary projector.")
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
