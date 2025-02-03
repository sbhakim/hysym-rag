# src/knowledge_integrator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlignmentLayer(nn.Module):
    """
    An enhanced alignment layer that dynamically bridges symbolic and neural representations.
    This implementation adds:
    1. Multi-head attention for better feature alignment
    2. Residual connections for stable training
    3. Layer normalization for better convergence
    4. Confidence scoring for alignment quality
    """

    def __init__(
            self,
            sym_dim: int = 300,
            neural_dim: int = 768,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        """
        Initialize the enhanced alignment layer.

        Args:
            sym_dim: Dimension of symbolic embeddings (default: 300 for typical word embeddings)
            neural_dim: Dimension of neural embeddings (default: 768 for BERT-based models)
            num_heads: Number of attention heads for multi-head attention
            dropout: Dropout rate for regularization
        """
        super(AlignmentLayer, self).__init__()

        # Ensure dimensions are compatible with multi-head attention
        self.head_dim = 64  # Standard dimension per head
        self.hidden_dim = self.head_dim * num_heads

        # Initial projection layers
        self.sym_projection = nn.Sequential(
            nn.Linear(sym_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.neural_projection = nn.Sequential(
            nn.Linear(neural_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-head attention components
        self.num_heads = num_heads
        self.scale = math.sqrt(self.head_dim)

        # Final alignment layers
        self.alignment_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, neural_dim)
        )

        # Confidence scoring module
        self.confidence_scorer = nn.Sequential(
            nn.Linear(neural_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def dynamic_project(
            self,
            sym_emb: torch.Tensor,
            neural_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically project symbolic embeddings to neural space using multi-head attention.

        Args:
            sym_emb: Symbolic embeddings [batch_size, sym_dim]
            neural_emb: Neural embeddings [batch_size, neural_dim]

        Returns:
            Tuple of (aligned_embedding, attention_weights)
        """
        # Initial projections
        sym_projected = self.sym_projection(sym_emb)
        neural_projected = self.neural_projection(neural_emb)

        # Reshape for multi-head attention
        batch_size = sym_projected.size(0)
        sym_heads = sym_projected.view(batch_size, -1, self.num_heads, self.head_dim)
        neural_heads = neural_projected.view(batch_size, -1, self.num_heads, self.head_dim)

        # Calculate attention scores
        attention_scores = torch.matmul(sym_heads, neural_heads.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended_features = torch.matmul(attention_weights, neural_heads)
        attended_features = attended_features.view(batch_size, -1, self.hidden_dim)

        return attended_features, attention_weights

    def forward(
            self,
            sym_emb: torch.Tensor,
            neural_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, float, Optional[dict]]:
        """
        Perform the full alignment process with confidence scoring.

        Args:
            sym_emb: Symbolic embeddings
            neural_emb: Neural embeddings

        Returns:
            Tuple of (aligned_embedding, confidence_score, debug_info)
        """
        try:
            # Ensure tensors are on the same device
            if sym_emb.device != neural_emb.device:
                sym_emb = sym_emb.to(neural_emb.device)

            # Dynamic projection
            attended_features, attention_weights = self.dynamic_project(sym_emb, neural_emb)

            # Concatenate with original neural embeddings for residual connection
            combined = torch.cat([attended_features, neural_emb.unsqueeze(1)], dim=-1)

            # Final alignment projection
            aligned_embedding = self.alignment_projection(combined)

            # Calculate confidence score
            confidence_score = self.confidence_scorer(aligned_embedding).squeeze(-1)

            # Prepare debug information
            debug_info = {
                'attention_weights': attention_weights.detach(),
                'confidence_score': confidence_score.detach(),
                'alignment_magnitude': torch.norm(aligned_embedding).item()
            }

            logger.debug(f"Alignment completed successfully. Confidence: {confidence_score.mean().item():.4f}")

            return aligned_embedding, confidence_score.mean().item(), debug_info

        except Exception as e:
            logger.error(f"Error in alignment process: {str(e)}")
            # Return original neural embedding as fallback
            return neural_emb, 0.0, None

    def compute_loss(
            self,
            aligned_emb: torch.Tensor,
            target_emb: torch.Tensor,
            confidence_score: float
    ) -> torch.Tensor:
        """
        Compute the alignment loss with confidence weighting.

        Args:
            aligned_emb: The aligned embeddings
            target_emb: The target neural embeddings
            confidence_score: The confidence score for this alignment

        Returns:
            Weighted loss value
        """
        # Cosine similarity loss
        cos_loss = 1 - F.cosine_similarity(aligned_emb, target_emb, dim=-1).mean()

        # L2 regularization
        l2_loss = torch.norm(aligned_emb - target_emb, p=2, dim=-1).mean()

        # Combine losses with confidence weighting
        combined_loss = (0.7 * cos_loss + 0.3 * l2_loss) * confidence_score

        return combined_loss