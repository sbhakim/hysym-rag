#src/knowledge_integrator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Optional, List
from src.utils.device_manager import DeviceManager
from src.utils.dimension_manager import DimensionalityManager

#Configure logging to DEBUG level to capture detailed logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class AlignmentLayer(nn.Module):
    def __init__(
        self,
        sym_dim: int,  # Corrected: Added to __init__ signature
        neural_dim: int,  # Corrected: Added to __init__ signature
        target_dim: int = 768,
        num_heads: int = 4,  # unused right now.
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dim_manager: Optional[DimensionalityManager] = None
        ):
        """
        Enhanced alignment layer with dynamic dimension handling, confidence-weighted fusion,
        and reasoning chain tracking for academic evaluation.
        """
        super().__init__() # Corrected: Use super().__init__() to properly initialize nn.Module
        self.device = device if device is not None else DeviceManager.get_device()
        self.head_dim = 64
        self.num_heads = num_heads  # currently unused
        self.hidden_dim = self.head_dim * num_heads
        self.target_dim = target_dim
        self.dim_manager = dim_manager or DimensionalityManager(target_dim=target_dim, device=device)

        # New projection adapters: project from target_dim (768) to hidden_dim (256)
        self.sym_projection_adapter = nn.Linear(target_dim, self.hidden_dim)
        self.neural_projection_adapter = nn.Linear(target_dim, self.hidden_dim)

        # Main projection layers for each branch (processing in hidden_dim space)
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

        # Final alignment projection: combine features from both branches
        self.alignment_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Combine attended features and neural projection
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, target_dim)
        )

        # Confidence scoring modules
        self.confidence_scorer = nn.Sequential(
            nn.Linear(target_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.rule_confidence_gate = nn.Sequential(
            nn.Linear(target_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.context_analyzer = nn.Sequential(
            nn.Linear(target_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Initialize chain tracking metrics for academic evaluation.
        # Stores lists of chain lengths, average confidence scores, and computed inference depths.
        self.chain_metrics = {
            'lengths': [],
            'confidences': [],
            'depths': []
        }

    def _validate_and_prepare_input(self, emb: torch.Tensor, name: str) -> torch.Tensor:
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        logger.debug(f"{name} embedding shape: {emb.shape}")
        return emb

    def dynamic_project(self, sym_emb: torch.Tensor, neural_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # No need for re-alignment here, use input embeddings directly.
        try:
            batch_size = sym_emb.size(0)

            # Project the aligned embeddings to the hidden dimension via dedicated adapters
            sym_proj_input = self.sym_projection_adapter(sym_emb)
            neural_proj_input = self.neural_projection_adapter(neural_emb)

            # Process through branch-specific projection layers
            sym_projected = self.sym_projection(sym_proj_input)
            neural_projected = self.neural_projection(neural_proj_input)
            logger.debug(f"Symbolic Projected Shape: {sym_projected.shape}")  # Expected: [batch, hidden_dim]
            logger.debug(f"Neural Projected Shape: {neural_projected.shape}")  # Expected: [batch, hidden_dim]

            # Reshape for multi-head attention: [batch, tokens, num_heads, head_dim]
            # (Assuming single token for now)
            sym_heads = sym_projected.view(batch_size, -1, self.num_heads, self.head_dim)
            neural_heads = neural_projected.view(batch_size, -1, self.num_heads, self.head_dim)
            logger.debug(f"Symbolic Heads Shape: {sym_heads.shape}")  # Expected: [batch, 1, num_heads, head_dim]
            logger.debug(f"Neural Heads Shape: {neural_heads.shape}")  # Expected: [batch, 1, num_heads, head_dim]

            # Compute attention scores and weights. Key matrix multiplication happens here.
            attention_scores = torch.matmul(sym_heads, neural_heads.transpose(-2, -1)) / self.scale
            logger.debug(f"Attention Scores shape: {attention_scores.shape}")  # Expected: [batch, num_heads, 1, 1]
            attention_weights = F.softmax(attention_scores, dim=-1)
            attended_features = torch.matmul(attention_weights,
                                            neural_heads)  # Expected output: [batch, num_heads, 1, head_dim]
            attended_features = attended_features.view(batch_size, -1,
                                                    self.hidden_dim)  # Expected output: [batch, 1, hidden_dim]
            logger.debug(f"Attended features shape: {attended_features.shape}")  # Expected: [batch, 1, hidden_dim]

            return attended_features, attention_weights

        except Exception as e:
            logger.error(f"Error in dynamic_project: {str(e)}")
            raise

    def traverse_graph_from_multi_hop(self, embeddings: torch.Tensor) -> List[str]:
        """
        Dummy multi-hop traversal: for each hop (dimension 1 of embeddings), return a fixed symbolic response.
        This is a placeholder and should be replaced with actual multi-hop logic.
        """
        if embeddings.dim() < 2:
            return ["Symbolic response"]
        hop_count = embeddings.size(1)
        return ["Symbolic response" for _ in range(hop_count)]

    def _track_reasoning_chain(self, query_embedding: torch.Tensor, response_embedding: torch.Tensor,
                            chain_info: Optional[dict] = None) -> dict:
        """
        Track reasoning chain metrics for academic analysis.
        Returns a dictionary with 'chain_length', 'confidence', and 'depth'.
        """
        metrics = {
            'chain_length': 0,
            'confidence': 0.0,
            'depth': 0
        }
        try:
            if chain_info and 'steps' in chain_info:
                metrics['chain_length'] = len(chain_info['steps'])
            # Use cosine similarity between query and response embeddings to derive a confidence score.
            if query_embedding is not None and response_embedding is not None:
                similarity = F.cosine_similarity(query_embedding.view(1, -1), response_embedding.view(1, -1)).item()
                metrics['confidence'] = max(0.0, min(1.0, similarity))
            if chain_info and 'reasoning_path' in chain_info:
                metrics['depth'] = len(chain_info['reasoning_path'])
            # Update internal tracking
            self.chain_metrics['lengths'].append(metrics['chain_length'])
            self.chain_metrics['confidences'].append(metrics['confidence'])
            self.chain_metrics['depths'].append(metrics['depth'])
        except Exception as e:
            logger.error(f"Error tracking reasoning chain: {str(e)}")
        return metrics

    def forward(self, sym_emb: torch.Tensor, neural_emb: torch.Tensor, rule_confidence: Optional[float] = None) -> \
            Tuple[torch.Tensor, float, Optional[dict]]:
        try:
            self._log_memory_usage("start_forward")
            sym_emb = self._validate_and_prepare_input(sym_emb, "Symbolic")
            neural_emb = self._validate_and_prepare_input(neural_emb, "Neural")

            sym_emb = self._ensure_device(sym_emb, neural_emb.device)
            self._log_memory_usage("after_device_placement")

            attended_features, attention_weights = self.dynamic_project(sym_emb, neural_emb)
            self._log_memory_usage("after_projection")

            if attended_features.size(-1) != self.hidden_dim:
                logger.warning(
                    f"Attended features dimension mismatch: {attended_features.size(-1)} vs {self.hidden_dim}")

            attended_features = self._ensure_device(attended_features, neural_emb.device)
            neural_emb = self._ensure_device(neural_emb, attended_features.device)
            # Project neural embedding to hidden_dim space:
            neural_proj = self.neural_projection_adapter(neural_emb)
            neural_proj = self.neural_projection(neural_proj)
            neural_proj = neural_proj.unsqueeze(1)  # Shape: [batch, 1, hidden_dim]

            # Concatenate attended features with the projected neural embedding
            combined = torch.cat([attended_features, neural_proj], dim=-1)
            self._log_memory_usage("after_combination")

            aligned_embedding = self.alignment_projection(combined)
            context_score = self.context_analyzer(aligned_embedding)
            rule_score = self.rule_confidence_gate(aligned_embedding)  # Get rule_score

            # Tuned confidence calculation (incorporating rule_confidence)
            base_confidence = (
                                        context_score + rule_score) / 2  # Base confidence from context and rule gates
            fusion_confidence = torch.clamp(base_confidence + 0.2, 0.3, 0.9)  # Apply clamp and bias
            if rule_confidence is not None:
                fusion_confidence = fusion_confidence + rule_confidence * 0.1  # Rule context boost

            confidence_score = fusion_confidence  # Use fusion_confidence

            # Apply a minimum confidence threshold of 0.1 to avoid zero values.
            MIN_CONFIDENCE = 0.1
            confidence_score = torch.max(confidence_score, torch.tensor(MIN_CONFIDENCE, device=confidence_score.device))

            self._log_memory_usage("after_alignment")

            fused_embedding = confidence_score.unsqueeze(-1) * aligned_embedding + (
                        1 - confidence_score.unsqueeze(-1)) * neural_emb.unsqueeze(1)

            # --- Reasoning Chain Metrics Tracking ---
            # Create a basic chain_info dictionary using a dummy traversal.
            chain_info = {
                'steps': self.traverse_graph_from_multi_hop(neural_emb.unsqueeze(0).unsqueeze(0)),
                'reasoning_path': self.traverse_graph_from_multi_hop(neural_emb.unsqueeze(0).unsqueeze(0))
            }
            chain_metrics = self._track_reasoning_chain(sym_emb, aligned_embedding, chain_info=chain_info)
            chain_length = chain_metrics.get('chain_length', 1)
            chain_conf = chain_metrics.get('confidence', 0.0)
            inference_depth = chain_metrics.get('depth', 1)

            debug_info = {
                'attention_weights': attention_weights.detach().cpu().numpy(),
                'confidence_score': chain_conf,
                'alignment_magnitude': torch.norm(aligned_embedding).item(),
                'sym_emb_shape': list(sym_emb.shape),
                'neural_emb_shape': list(neural_emb.shape),
                'aligned_emb_shape': list(aligned_embedding.shape),
                'fused_emb_shape': list(fused_embedding.shape),
                'attended_features_shape': list(attended_features.shape),
                'combined_shape': list(combined.shape),
                'context_score': context_score.mean().item(),
                'chain_length': chain_length,
                'inference_depth': inference_depth,
                'chain_confidence': chain_conf,
                'rule_score': rule_score.mean().item(),  # Include rule_score in debug info
                'base_confidence': base_confidence.mean().item(),  # Include base_confidence
                'fusion_confidence': fusion_confidence.mean().item()  # Include fusion_confidence

            }
            if rule_confidence is not None:
                debug_info['rule_confidence_input'] = rule_confidence  # Include rule_confidence_input if available

            logger.info(f"Alignment completed successfully. Confidence: {chain_conf:.4f}")
            self._log_memory_usage("end_forward")
            return fused_embedding, chain_conf, debug_info

        except Exception as e:
            logger.error(f"Unexpected error in forward pass: {str(e)}")
            self._log_memory_usage("error_state")
            # Fallback: return neural embedding, 0 confidence, and the error info.
            return neural_emb, 0.0, {'error': str(e), 'fallback': 'using_neural_embedding'}

    def compute_loss(self, aligned_emb: torch.Tensor, target_emb: torch.Tensor, confidence_score: float,
                     lambda_cos: float = 0.7, lambda_l2: float = 0.3) -> torch.Tensor:
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
        moved_tensor, _ = DeviceManager.ensure_same_device(tensor, tensor, device=device)
        return moved_tensor