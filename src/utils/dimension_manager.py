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
    Provides caching and validation of dimension transformations with adaptive projection and error recovery.
    """

    def __init__(self, target_dim: int = 768, device: Optional[torch.device] = None):
        self.target_dim = target_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Registry for adapters keyed by a string of source and input_dim
        self.adapter_registry: Dict[str, nn.Module] = {}
        self.mismatch_counts: Dict[str, int] = defaultdict(int)
        # Validation layer for explicit 384 to 768 conversion
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
        self.adapter_registry[name] = adapter.to(self.device)
        logger.info(f"Adapter registered for {name} with target_dim {self.target_dim}")

    def get_adapter(self, source: str, current_dim: int) -> nn.Module:
        """
        Returns a linear adapter from current_dim to target_dim.
        If one does not exist for the given source, create and register one.
        """
        key = f"{source}_{current_dim}_to_{self.target_dim}"
        if key not in self.adapter_registry:
            adapter = nn.Linear(current_dim, self.target_dim).to(self.device)
            self.adapter_registry[key] = adapter
            logger.info(f"[DEBUG] Created new adapter for {source}: {current_dim} -> {self.target_dim}")
        return self.adapter_registry[key]

    def align_and_log(self, tensor: torch.Tensor, source: str) -> torch.Tensor:
        """
        Align tensor to target_dim if necessary.
        Logs the dimensions before and after alignment for easier debugging.
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        original_shape = tensor.shape
        logger.debug(f"[DEBUG] {source} tensor original shape: {original_shape}")
        if tensor.size(-1) == self.target_dim:
            logger.debug(f"[DEBUG] {source} tensor already aligned: {original_shape}")
            return tensor.to(self.device)
        if tensor.size(-1) == 384:
            logger.debug(f"[DEBUG] Aligning {source} tensor from 384 to {self.target_dim} using validation_layer")
            aligned_tensor = self.validation_layer(tensor.to(self.device))
            logger.debug(f"[DEBUG] {source} tensor aligned: {original_shape} -> {aligned_tensor.shape}")
            return aligned_tensor
        adapter = self.get_adapter(source, tensor.size(-1))
        aligned_tensor = adapter(tensor.to(self.device))
        logger.debug(f"[DEBUG] {source} tensor aligned using adapter: {original_shape} -> {aligned_tensor.shape}")
        return aligned_tensor

    def align_embeddings(self,
                         embedding: torch.Tensor,
                         source: str,
                         return_confidence: bool = False,
                         already_aligned: bool = False) -> torch.Tensor:
        """
        Enhanced embedding alignment.
        If already_aligned is True, returns the tensor immediately.
        Otherwise, delegates to align_and_log for consistent alignment.
        If return_confidence is True, returns a tuple (aligned_embedding, confidence).
        """
        try:
            if already_aligned:
                logger.debug(f"Embedding from source '{source}' is marked as already aligned; skipping alignment.")
                return (embedding, 1.0) if return_confidence else embedding
            aligned = self.align_and_log(embedding, source)
            if return_confidence:
                confidence = self._calculate_alignment_confidence(embedding, aligned)
                logger.debug(f"[DEBUG] Alignment confidence for source '{source}': {confidence:.4f}")
                return aligned, confidence
            return aligned
        except Exception as e:
            logger.error(f"Error in alignment for source '{source}': {str(e)}")
            raise

    def _create_projection(self, input_dim: int) -> nn.Module:
        """
        Creates a new projection layer to map embeddings from input_dim to target_dim with adaptive dropout.
        """
        projection = nn.Sequential(
            nn.Linear(input_dim, self.target_dim),
            nn.LayerNorm(self.target_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        with torch.no_grad():
            std = (2.0 / (input_dim + self.target_dim)) ** 0.5
            projection[0].weight.data.normal_(0.0, std)
        return projection.to(self.device)

    def _calculate_alignment_confidence(self,
                                        original: torch.Tensor,
                                        aligned: torch.Tensor) -> float:
        """
        Calculates a confidence score for the alignment based on cosine similarity.
        """
        with torch.no_grad():
            if original.size(-1) != aligned.size(-1):
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
            "total_adapters": len(self.adapter_registry),
            "mismatch_counts": dict(self.mismatch_counts)
        }
