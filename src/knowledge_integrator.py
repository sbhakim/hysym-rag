# src/knowledge_integrator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DimensionAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.projection = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input dimension does not match expected, warn and adjust
        if x.size(-1) != self.in_features:
            logger.warning(f"Input dimension mismatch: {x.size(-1)} vs expected {self.in_features}. Adjusting...")
            # Create a temporary linear layer to project to the expected dimension
            adjust_layer = nn.Linear(x.size(-1), self.in_features).to(x.device)
            x = adjust_layer(x)
        projected = self.projection(x)
        return self.norm(projected)

class AlignmentLayer(nn.Module):
    def __init__(
            self,
            sym_dim: int = 300,
            neural_dim: int = 768,
            target_dim: int = 768,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        """
        Enhanced alignment layer with dynamic dimension handling and confidence-weighted fusion.
        """
        super(AlignmentLayer, self).__init__()

        self.head_dim = 64
        self.num_heads = num_heads
        self.hidden_dim = self.head_dim * num_heads
        self.target_dim = target_dim

        # Dimension adapters for input embeddings
        self.sym_adapter = DimensionAdapter(sym_dim, self.hidden_dim)
        self.neural_adapter = DimensionAdapter(neural_dim, self.hidden_dim)

        # Main projection layers for each branch
        self.sym_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.neural_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.scale = math.sqrt(self.head_dim)

        # Final alignment projection: combining features from both branches
        self.alignment_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, target_dim)
        )

        # Confidence scoring module (original)
        self.confidence_scorer = nn.Sequential(
            nn.Linear(target_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # NEW: Rule confidence gate for additional rule-based confidence estimation
        self.rule_confidence_gate = nn.Sequential(
            nn.Linear(target_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # NEW: Context analyzer for computing context-based confidence
        self.context_analyzer = nn.Sequential(
            nn.Linear(target_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _validate_and_prepare_input(self, emb: torch.Tensor, name: str) -> torch.Tensor:
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        logger.debug(f"{name} embedding shape: {emb.shape}")
        return emb

    def dynamic_project(self, sym_emb: torch.Tensor, neural_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            device = sym_emb.device
            sym_emb = self.sym_adapter(sym_emb.to(device))
            neural_emb = self.neural_adapter(neural_emb.to(device))
            sym_projected = self.sym_projection(sym_emb)
            neural_projected = self.neural_projection(neural_emb)
            batch_size = sym_projected.size(0)
            sym_heads = sym_projected.view(batch_size, -1, self.num_heads, self.head_dim)
            neural_heads = neural_projected.view(batch_size, -1, self.num_heads, self.head_dim)
            attention_scores = torch.matmul(sym_heads, neural_heads.transpose(-2, -1)) / self.scale
            attention_weights = F.softmax(attention_scores, dim=-1)
            attended_features = torch.matmul(attention_weights, neural_heads)
            attended_features = attended_features.view(batch_size, -1, self.hidden_dim)
            return attended_features, attention_weights
        except Exception as e:
            logger.error(f"Error in dynamic_project: {str(e)}")
            raise

    def forward(self, sym_emb: torch.Tensor, neural_emb: torch.Tensor, rule_confidence: Optional[float] = None) -> Tuple[torch.Tensor, float, Optional[dict]]:
        try:
            self._log_memory_usage("start_forward")
            sym_emb = self._validate_and_prepare_input(sym_emb, "Symbolic")
            neural_emb = self._validate_and_prepare_input(neural_emb, "Neural")
            if sym_emb.size(1) != self.sym_adapter.in_features:
                raise ValueError(f"Symbolic embedding size mismatch: {sym_emb.size(1)} vs expected {self.sym_adapter.in_features}")
            if neural_emb.size(1) != self.neural_adapter.in_features:
                raise ValueError(f"Neural embedding size mismatch: {neural_emb.size(1)} vs expected {self.neural_adapter.in_features}")
            sym_emb = self._ensure_device(sym_emb, neural_emb.device)
            self._log_memory_usage("after_device_placement")
            try:
                attended_features, attention_weights = self.dynamic_project(sym_emb, neural_emb)
                self._log_memory_usage("after_projection")
            except RuntimeError as e:
                logger.error(f"Error in dynamic projection: {str(e)}")
                sym_emb = self.sym_adapter(sym_emb)
                neural_emb = self.neural_adapter(neural_emb)
                attended_features, attention_weights = self.dynamic_project(sym_emb, neural_emb)
            if attended_features.size(-1) != self.hidden_dim:
                logger.warning(f"Attended features dimension mismatch: {attended_features.size(-1)} vs {self.hidden_dim}")
                attended_features = self.sym_adapter(attended_features)
            try:
                attended_features = self._ensure_device(attended_features, neural_emb.device)
                neural_emb = self._ensure_device(neural_emb, attended_features.device)
                combined = torch.cat([attended_features, neural_emb.unsqueeze(1)], dim=-1)
                self._log_memory_usage("after_combination")
            except RuntimeError as e:
                logger.error(f"Error in feature combination: {str(e)}")
                if attended_features.size(-1) != neural_emb.size(-1):
                    neural_emb = self.neural_adapter(neural_emb)
                combined = torch.cat([attended_features, neural_emb.unsqueeze(1)], dim=-1)
            aligned_embedding = self.alignment_projection(combined)
            # Compute context-based confidence
            context_score = self.context_analyzer(aligned_embedding)
            # If an external rule_confidence value is provided, compute a rule-based score and average it
            if rule_confidence is not None:
                rule_score = self.rule_confidence_gate(aligned_embedding)
                dynamic_confidence = (context_score + rule_score) / 2
            else:
                dynamic_confidence = context_score
            # Alternatively, you might still combine with the original confidence_scorer if needed:
            # confidence_score = self.confidence_scorer(aligned_embedding).squeeze(-1)
            confidence_score = dynamic_confidence  # Using the dynamic confidence computed above
            self._log_memory_usage("after_alignment")
            # Dynamic fusion: fuse aligned and original neural embedding weighted by the computed confidence score.
            fused_embedding = confidence_score.unsqueeze(-1) * aligned_embedding + (1 - confidence_score.unsqueeze(-1)) * neural_emb.unsqueeze(1)
            debug_info = {
                'attention_weights': attention_weights.detach().cpu().numpy(),
                'confidence_score': confidence_score.mean().item(),
                'alignment_magnitude': torch.norm(aligned_embedding).item(),
                'sym_emb_shape': list(sym_emb.shape),
                'neural_emb_shape': list(neural_emb.shape),
                'aligned_emb_shape': list(aligned_embedding.shape),
                'fused_emb_shape': list(fused_embedding.shape),
                'attended_features_shape': list(attended_features.shape),
                'combined_shape': list(combined.shape),
                'context_score': context_score.mean().item()
            }
            if rule_confidence is not None:
                debug_info['rule_score'] = rule_score.mean().item()
            logger.info(f"Alignment completed successfully. Confidence: {confidence_score.mean().item():.4f}")
            self._log_memory_usage("end_forward")
            return fused_embedding, confidence_score.mean().item(), debug_info

        except ValueError as ve:
            logger.error(f"Validation error in forward pass: {str(ve)}")
            try:
                if hasattr(self, 'neural_adapter'):
                    neural_emb = self.neural_adapter(neural_emb)
                    return neural_emb, 0.0, {'error': str(ve), 'fallback': 'using_adapted_neural_embedding'}
            except:
                pass
            return neural_emb, 0.0, {'error': str(ve), 'fallback': 'using_neural_embedding'}
        except Exception as e:
            logger.error(f"Unexpected error in forward pass: {str(e)}")
            self._log_memory_usage("error_state")
            return neural_emb, 0.0, {'error': str(e), 'fallback': 'using_neural_embedding'}

    def compute_loss(self, aligned_emb: torch.Tensor, target_emb: torch.Tensor, confidence_score: float, lambda_cos: float = 0.7, lambda_l2: float = 0.3) -> torch.Tensor:
        try:
            cos_loss = 1 - F.cosine_similarity(aligned_emb, target_emb, dim=-1).mean()
            l2_loss = torch.norm(aligned_emb - target_emb, p=2, dim=-1).mean()
            combined_loss = (lambda_cos * cos_loss + lambda_l2 * l2_loss) * confidence_score
            return combined_loss
        except Exception as e:
            logger.error(f"Error in loss computation: {str(e)}")
            return torch.tensor(float('inf'), device=aligned_emb.device)

    def _log_memory_usage(self, stage: str):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Memory usage at {stage}: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    def _ensure_device(self, tensor: torch.Tensor, target_device: Optional[torch.device] = None) -> torch.Tensor:
        device = target_device or self.device
        return tensor.to(device) if tensor.device != device else tensor
