# src/knowledge_integrator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Optional, List
from src.utils.device_manager import DeviceManager
from src.utils.dimension_manager import DimensionalityManager
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AlignmentLayer(nn.Module):
    def __init__(
            self,
            sym_dim: int,
            neural_dim: int,
            target_dim: int = 768,
            num_heads: int = 4,
            dropout: float = 0.1,
            device: Optional[torch.device] = None,
            dim_manager: Optional[DimensionalityManager] = None,
            base_confidence_value: float = 0.3,
            similarity_weight: float = 0.5,
            context_weight: float = 0.2,
            rule_weight: float = 0.2,
            min_confidence: float = 0.5,
            max_confidence: float = 0.95
    ):
        super().__init__()
        self.device = device if device is not None else DeviceManager.get_device()
        self.head_dim = 64
        self.num_heads = num_heads
        self.hidden_dim = self.head_dim * num_heads
        self.target_dim = target_dim
        self.dim_manager = dim_manager or DimensionalityManager(target_dim=target_dim, device=self.device)
        self.base_confidence_value = base_confidence_value
        self.similarity_weight = similarity_weight
        self.context_weight = context_weight
        self.rule_weight = rule_weight
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.sym_projection_adapter = nn.Linear(target_dim, self.hidden_dim)
        self.neural_projection_adapter = nn.Linear(target_dim, self.hidden_dim)
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
        self.alignment_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, target_dim)
        )
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
        try:
            batch_size = sym_emb.size(0)
            sym_proj_input = self.sym_projection_adapter(sym_emb)
            neural_proj_input = self.neural_projection_adapter(neural_emb)
            sym_projected = self.sym_projection(sym_proj_input)
            neural_projected = self.neural_projection(neural_proj_input)
            logger.debug(f"Symbolic Projected Shape: {sym_projected.shape}")
            logger.debug(f"Neural Projected Shape: {neural_projected.shape}")
            sym_heads = sym_projected.reshape(batch_size, -1, self.num_heads, self.head_dim)
            neural_heads = neural_projected.reshape(batch_size, -1, self.num_heads, self.head_dim)
            logger.debug(f"Symbolic Heads Shape: {sym_heads.shape}")
            logger.debug(f"Neural Heads Shape: {neural_heads.shape}")
            attention_scores = torch.matmul(sym_heads, neural_heads.transpose(-2, -1)) / self.scale
            logger.debug(f"Attention Scores shape: {attention_scores.shape}")
            attention_weights = F.softmax(attention_scores, dim=-1)
            attended_features = torch.matmul(attention_weights, neural_heads)
            attended_features = attended_features.reshape(batch_size, -1, self.hidden_dim)
            logger.debug(f"Attended features shape: {attended_features.shape}")
            return attended_features, attention_weights
        except Exception as e:
            logger.error(f"Error in dynamic_project: {str(e)}")
            raise

    def traverse_graph_from_multi_hop(self, embeddings: torch.Tensor) -> List[str]:
        if embeddings.dim() < 2:
            return ["Symbolic response"]
        hop_count = embeddings.size(1)
        return ["Symbolic response" for _ in range(hop_count)]

    def _track_reasoning_chain(self, query_embedding: torch.Tensor, response_embedding: torch.Tensor,
                               chain_info: Optional[dict] = None) -> dict:
        metrics = {'chain_length': 0, 'confidence': 0.0, 'depth': 0}
        try:
            if chain_info and 'steps' in chain_info:
                metrics['chain_length'] = len(chain_info['steps'])
            if query_embedding is not None and response_embedding is not None:
                similarity = F.cosine_similarity(query_embedding.view(1, -1), response_embedding.view(1, -1)).item()
                metrics['confidence'] = max(0.0, min(1.0, similarity))
            if chain_info and 'reasoning_path' in chain_info:
                metrics['depth'] = len(chain_info['reasoning_path'])
            self.chain_metrics['lengths'].append(metrics['chain_length'])
            self.chain_metrics['confidences'].append(metrics['confidence'])
            self.chain_metrics['depths'].append(metrics['depth'])
        except Exception as e:
            logger.error(f"Error tracking reasoning chain: {str(e)}")
        return metrics

    def forward(self, sym_emb: torch.Tensor, neural_emb: torch.Tensor, rule_confidence: Optional[float] = None) -> \
            Tuple[torch.Tensor, float, Optional[dict]]:
        memory_tracking = os.environ.get('SYMRAG_MEMORY_TRACKING', '0') == '1'
        try:
            if memory_tracking:
                self._log_memory_usage("start_forward")
            sym_emb = self._validate_and_prepare_input(sym_emb, "Symbolic").to(self.device)
            neural_emb = self._validate_and_prepare_input(neural_emb, "Neural").to(self.device)
            if memory_tracking:
                self._log_memory_usage("after_device_placement")
            attended_features, attention_weights = self.dynamic_project(sym_emb, neural_emb)
            if memory_tracking:
                self._log_memory_usage("after_projection")
            if attended_features.size(-1) != self.hidden_dim:
                logger.warning(
                    f"Attended features dimension mismatch: {attended_features.size(-1)} vs {self.hidden_dim}")
            neural_proj = self.neural_projection_adapter(neural_emb)
            neural_proj = self.neural_projection(neural_proj)
            neural_proj = neural_proj.unsqueeze(1)
            combined = torch.cat([attended_features, neural_proj], dim=-1)
            if memory_tracking:
                self._log_memory_usage("after_combination")
            aligned_embedding = self.alignment_projection(combined)
            context_score = self.context_analyzer(aligned_embedding)
            rule_score = self.rule_confidence_gate(aligned_embedding)
            context_score_val = context_score.mean().item()
            rule_score_val = rule_score.mean().item()
            sim_score = F.cosine_similarity(sym_emb.view(1, -1), neural_emb.view(1, -1)).item()
            fusion_confidence_tensor = torch.tensor(
                self.base_confidence_value +
                (sim_score * self.similarity_weight) +
                (context_score_val * self.context_weight) +
                (rule_score_val * self.rule_weight),
                device=self.device
            )
            if rule_confidence is not None:
                fusion_confidence_tensor += rule_confidence * 0.1
            fusion_confidence = torch.clamp(fusion_confidence_tensor, self.min_confidence, self.max_confidence)
            confidence_score = fusion_confidence
            logger.info(f"Alignment confidence: sim_score={sim_score:.4f}, "
                        f"context_score={context_score_val:.4f}, rule_score={rule_score_val:.4f}, "
                        f"final_confidence={confidence_score.item():.4f}")
            if memory_tracking:
                self._log_memory_usage("after_alignment")
            fused_embedding = confidence_score.unsqueeze(-1) * aligned_embedding + (
                    1 - confidence_score.unsqueeze(-1)) * neural_emb.unsqueeze(1)
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
                'context_score': context_score_val,
                'chain_length': chain_length,
                'inference_depth': inference_depth,
                'chain_confidence': chain_conf,
                'rule_score': rule_score_val,
                'fusion_confidence': fusion_confidence.item()
            }
            if rule_confidence is not None:
                debug_info['rule_confidence_input'] = rule_confidence
            logger.info(f"Alignment completed. Confidence: {confidence_score.item():.4f}")
            if memory_tracking:
                self._log_memory_usage("end_forward")
            return fused_embedding, confidence_score.item(), debug_info
        except Exception as e:
            logger.error(f"Unexpected error in forward pass: {str(e)}")
            if memory_tracking:
                self._log_memory_usage("error_state")
            return neural_emb, 0.0, {'error': str(e), 'fallback': 'using_neural_embedding'}

    def compute_loss(self, aligned_emb: torch.Tensor, target_emb: torch.Tensor, confidence_score: float,
                     lambda_cos: float = 0.7, lambda_l2: float = 0.3) -> torch.Tensor:
        pass

    def _log_memory_usage(self, stage: str):
        if stage in ['start_forward', 'end_forward'] or os.environ.get('SYMRAG_MEMORY_TRACKING') == '1':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"Memory usage at {stage}: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    def _ensure_device(self, tensor: torch.Tensor, target_device: Optional[torch.device] = None) -> torch.Tensor:
        device = target_device or self.device
        moved_tensor, _ = DeviceManager.ensure_same_device(tensor, tensor, device=device)
        return moved_tensor