"""
Advanced Deep Learning Training Protocols, Loss Functions, and Knowledge Distillation for Tabular Data.

This module provides state-of-the-art tabular neural network training components:
1. `BinaryFocalLoss`:
   Focal loss with tunable alpha and gamma (alpha=0.25 or scale_pos_weight balanced, gamma=2.0)
   to effectively combat extreme class imbalance (11-15% delay prevalence in HSCM).
   Fully differentiable for both binary ground truth and continuous soft targets (e.g. from mixup).
2. `CosineAnnealingWarmupRestarts`:
   Learning rate scheduler with initial linear warmup epochs to stabilize deep tabular training
   before cosine annealing decay and periodic restarts.
3. `TabularMixup`:
   Data augmentation engine for continuous and tabular representations generating synthetic
   convex combinations: x_mix = lambda * x_i + (1 - lambda) * x_j, y_mix = lambda * y_i + (1 - lambda) * y_j.
4. `TabularKnowledgeDistillationTrainer`:
   Teacher-student knowledge distillation training harness. Transfers "dark knowledge"
   and soft probability targets from top-performing GBDT ensembles (LightGBM, XGBoost, CatBoost)
   into tabular neural architectures (FT-Transformer, TabResNet) using a blended loss:
     Loss = alpha_distill * BCE(p_student, p_teacher) + (1 - alpha_distill) * FocalLoss(p_student, y_true)
5. `FTTransformer`:
   Feature Tokenizer Transformer for tabular data (Gorishniy et al., NeurIPS 2021) featuring
   continuous feature tokenization, entity embeddings, [CLS] token pooling, and multi-head self-attention.
"""

from __future__ import annotations

import math
import logging
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # Backwards compatibility

try:
    from src.utils.device_manager import get_optimal_device
    _DEFAULT_DEVICE, _DEVICE_INFO = get_optimal_device(prefer_cuda=True)
except Exception:
    _DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _DEVICE_INFO = {"device_type": _DEFAULT_DEVICE.type}

try:
    from src.config import MODELS_DIR
except Exception:
    MODELS_DIR = Path("results/models")

logger = logging.getLogger("DLTrainer")


# =============================================================================
# 1. Binary Focal Loss
# =============================================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with tunable alpha and gamma parameters.
    
    Addresses extreme class imbalance (such as 11-15% delays in pharmaceutical delivery data)
    by dynamically down-weighting well-classified easy negative examples and concentrating
    gradients on hard, minority-class delay instances.
    
    Mathematical Formulation:
      p_t = y * p + (1 - y) * (1 - p)
      alpha_t = y * alpha + (1 - y) * (1 - alpha)
      FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
      
    For soft targets y in [0, 1] (e.g., from TabularMixup or label smoothing):
      L_pos = - alpha * (1 - p)^gamma * log(p + eps)
      L_neg = - (1 - alpha) * p^gamma * log(1 - p + eps)
      Loss = y * L_pos + (1 - y) * L_neg
      
    Args:
        alpha: Weighting factor for positive class in (0, 1), default 0.25 (Lin et al.).
               If None and pos_weight is provided, alpha = pos_weight / (1.0 + pos_weight).
        gamma: Focusing parameter >= 0, default 2.0. Higher values reduce the relative loss
               for well-classified examples, putting more focus on hard misclassified samples.
        pos_weight: Optional positive-to-negative ratio weight (e.g. 6.0 ~ (1-0.14)/0.14).
                    Used to automatically compute balanced alpha if alpha is not set explicitly.
        eps: Small epsilon value for numerical stability avoiding log(0).
        reduction: 'mean' (default), 'sum', or 'none'.
        from_logits: If True, inputs are treated as unscaled logits and sigmoid is applied.
    """

    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[float] = None,
        eps: float = 1e-7,
        reduction: str = "mean",
        from_logits: bool = False,
    ):
        super().__init__()
        if pos_weight is not None and (alpha is None or alpha == 0.25):
            # Scale pos_weight into alpha in [0, 1]
            self.alpha = float(pos_weight) / (1.0 + float(pos_weight))
        elif alpha is not None:
            self.alpha = float(alpha)
        else:
            self.alpha = 0.25

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'. Must be 'mean', 'sum', or 'none'.")

        self.gamma = float(gamma)
        self.eps = float(eps)
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates Binary Focal Loss.
        
        Args:
            y_pred: Predicted probabilities in [0, 1] (or raw logits if from_logits=True).
                    Shape: (B,) or (B, 1) or (B, ...).
            y_true: Ground truth binary or continuous targets in [0, 1].
                    Shape matches y_pred or can be reshaped.
                    
        Returns:
            Scalar loss (or unreduced tensor if reduction='none').
        """
        # Align target dimensions and dtype
        if y_true.shape != y_pred.shape:
            y_true = y_true.view_as(y_pred)
        y_true = y_true.to(dtype=y_pred.dtype, device=y_pred.device)

        if self.from_logits:
            p = torch.sigmoid(y_pred)
        else:
            p = y_pred

        # Clamp probabilities for stable logarithm computation
        p = torch.clamp(p, self.eps, 1.0 - self.eps)

        # Compute positive and negative focal loss terms (smooth and continuous for mixup)
        # Positive component: when y is high, penalize low p
        loss_pos = - self.alpha * torch.pow(1.0 - p, self.gamma) * torch.log(p)
        # Negative component: when y is low, penalize high p
        loss_neg = - (1.0 - self.alpha) * torch.pow(p, self.gamma) * torch.log(1.0 - p)

        # Convex combination for continuous soft targets (e.g. TabularMixup) or binary targets
        loss = y_true * loss_pos + (1.0 - y_true) * loss_neg

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.4f}, gamma={self.gamma:.2f}, reduction='{self.reduction}'"


# =============================================================================
# 2. Cosine Annealing with Linear Warmup and Restarts Scheduler
# =============================================================================

class CosineAnnealingWarmupRestarts(LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Initial Linear Warmup and Restarts.
    
    Stabilizes deep tabular neural training by starting with small learning rates to prevent
    divergent gradient updates in early batch normalization and entity embeddings, linearly
    ramping up to `max_lr` over `warmup_steps`, and then executing smooth cosine annealing
    decay down to `min_lr`.
    
    Supports multi-cycle restarts with cycle length multipliers (`cycle_mult`) and peak
    learning rate decay (`gamma`).
    
    Args:
        optimizer: Wrapped PyTorch optimizer.
        first_cycle_steps: Number of steps (or epochs) in the first cosine cycle.
        cycle_mult: Factor to lengthen subsequent cycles on restart (default 1.0 = constant length).
        max_lr: Peak learning rate after warmup. If float, applied to all param groups.
                If None, initial lr from optimizer's param_groups is used.
        min_lr: Minimum floor learning rate at the trough of cosine cycle (default 1e-6).
        warmup_steps: Number of steps/epochs for linear warmup from min_lr to max_lr.
        gamma: Multiplicative factor of max_lr decay on each restart cycle (default 1.0).
        last_epoch: Index of last step/epoch (default -1).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: Optional[Union[float, List[float]]] = None,
        min_lr: Union[float, List[float]] = 1e-6,
        warmup_steps: int = 5,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        if first_cycle_steps <= 0:
            raise ValueError(f"first_cycle_steps must be > 0, got {first_cycle_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if warmup_steps >= first_cycle_steps:
            warmup_steps = max(0, first_cycle_steps - 1)

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = float(cycle_mult)
        self.warmup_steps = int(warmup_steps)
        self.gamma = float(gamma)

        num_groups = len(optimizer.param_groups)

        # Resolve max_lr per param group
        if max_lr is None:
            self.base_max_lrs = [group["lr"] for group in optimizer.param_groups]
        elif isinstance(max_lr, (int, float)):
            self.base_max_lrs = [float(max_lr)] * num_groups
        else:
            self.base_max_lrs = [float(lr) for lr in max_lr]

        # Resolve min_lr per param group
        if isinstance(min_lr, (int, float)):
            self.min_lrs = [float(min_lr)] * num_groups
        else:
            self.min_lrs = [float(lr) for lr in min_lr]

        self.current_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = -1 if last_epoch == -1 else last_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculates learning rates for the current step."""
        if not self._get_lr_called_within_step:
            logger.debug("get_lr() called directly rather than via scheduler.step().")

        lrs = []
        for base_max, min_l in zip(self.base_max_lrs, self.min_lrs):
            # Scale peak lr by decay factor gamma^cycle
            current_max = min_l + (base_max - min_l) * (self.gamma ** self.cycle)

            if self.step_in_cycle < self.warmup_steps:
                # 1. Linear Warmup Phase: min_lr -> current_max
                if self.warmup_steps == 0:
                    lr = current_max
                else:
                    progress = float(self.step_in_cycle) / float(self.warmup_steps)
                    lr = min_l + progress * (current_max - min_l)
            else:
                # 2. Cosine Annealing Phase: current_max -> min_lr
                anneal_steps = self.current_cycle_steps - self.warmup_steps
                progress = float(self.step_in_cycle - self.warmup_steps) / float(max(anneal_steps, 1))
                progress = min(max(progress, 0.0), 1.0)
                lr = min_l + 0.5 * (current_max - min_l) * (1.0 + math.cos(math.pi * progress))

            lrs.append(lr)
        return lrs

    def step(self, epoch: Optional[int] = None):
        """Advances scheduler by one step/epoch, tracking cycle boundaries."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.current_cycle_steps:
                # Transition to next cycle
                self.cycle += 1
                self.step_in_cycle = 0
                self.current_cycle_steps = int(self.current_cycle_steps * self.cycle_mult)
        else:
            # Reconstruct cycle from absolute epoch index
            self.last_epoch = epoch
            self.step_in_cycle = epoch
            self.cycle = 0
            self.current_cycle_steps = self.first_cycle_steps
            while self.step_in_cycle >= self.current_cycle_steps:
                self.step_in_cycle -= self.current_cycle_steps
                self.cycle += 1
                self.current_cycle_steps = int(self.current_cycle_steps * self.cycle_mult)

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            "first_cycle_steps": self.first_cycle_steps,
            "cycle_mult": self.cycle_mult,
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma,
            "base_max_lrs": self.base_max_lrs,
            "min_lrs": self.min_lrs,
            "current_cycle_steps": self.current_cycle_steps,
            "cycle": self.cycle,
            "step_in_cycle": self.step_in_cycle,
        })
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.first_cycle_steps = state_dict.pop("first_cycle_steps", self.first_cycle_steps)
        self.cycle_mult = state_dict.pop("cycle_mult", self.cycle_mult)
        self.warmup_steps = state_dict.pop("warmup_steps", self.warmup_steps)
        self.gamma = state_dict.pop("gamma", self.gamma)
        self.base_max_lrs = state_dict.pop("base_max_lrs", self.base_max_lrs)
        self.min_lrs = state_dict.pop("min_lrs", self.min_lrs)
        self.current_cycle_steps = state_dict.pop("current_cycle_steps", self.current_cycle_steps)
        self.cycle = state_dict.pop("cycle", self.cycle)
        self.step_in_cycle = state_dict.pop("step_in_cycle", self.step_in_cycle)
        super().load_state_dict(state_dict)


# =============================================================================
# 3. Tabular Mixup Data Augmentation
# =============================================================================

class TabularMixup(nn.Module):
    """
    Tabular Mixup Data Augmentation.
    
    Generates synthetic interpolated samples within mini-batches to regularize tabular neural
    networks and prevent overfitting on low-density regions of the feature manifold:
      x_mix = lambda * x_i + (1 - lambda) * x_j
      y_mix = lambda * y_i + (1 - lambda) * y_j
      
    When training with Knowledge Distillation, teacher soft probabilities are also interpolated:
      p_teacher_mix = lambda * p_teacher_i + (1 - lambda) * p_teacher_j
      
    Args:
        alpha: Beta distribution parameter Beta(alpha, alpha). If <= 0, mixup is disabled (lambda=1.0).
               Standard values for tabular data are 0.1 to 0.4. Default 0.2.
        prob: Probability of applying mixup to a given mini-batch in [0, 1]. Default 0.5.
        num_continuous: Optional number of leading columns that are continuous. If specified,
                        continuous features are linearly interpolated, while categorical features
                        (remaining columns) are sampled from parent i or j with probability lambda.
                        If None, all columns are treated as continuous (e.g. post-OHE/standardized).
        symmetric: If True, enforces lambda = max(lambda, 1 - lambda) so x_mix stays closer to x_i.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 0.5,
        num_continuous: Optional[int] = None,
        symmetric: bool = False,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.prob = float(prob)
        self.num_continuous = num_continuous
        self.symmetric = symmetric

    def sample_lambda(self, batch_size: int, device: torch.device) -> float:
        """Samples lambda from Beta(alpha, alpha)."""
        if self.alpha <= 0.0:
            return 1.0
        lam = float(np.random.beta(self.alpha, self.alpha))
        if self.symmetric:
            lam = max(lam, 1.0 - lam)
        return lam

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float]:
        """
        Applies Tabular Mixup to a mini-batch.
        
        Args:
            x: Input feature tensor of shape (B, D).
            y: Ground truth targets of shape (B,) or (B, 1).
            y_teacher: Optional teacher soft probability predictions of shape (B,) or (B, 1).
            
        Returns:
            Tuple of:
              - x_mixed: Interpolated feature tensor of shape (B, D).
              - y_mixed: Interpolated ground truth label tensor of shape (B, 1).
              - y_teacher_mixed: Interpolated teacher soft target (or None if y_teacher is None).
              - lam: Interpolation weight lambda in [0, 1].
        """
        # Ensure target shape is (B, 1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if y_teacher is not None and y_teacher.dim() == 1:
            y_teacher = y_teacher.unsqueeze(-1)

        batch_size = x.size(0)
        if not self.training or np.random.rand() > self.prob or batch_size <= 1:
            return x, y, y_teacher, 1.0

        lam = self.sample_lambda(batch_size, x.device)
        perm = torch.randperm(batch_size, device=x.device)

        # Mix features
        if self.num_continuous is not None and self.num_continuous < x.size(1):
            # Split continuous and categorical columns
            c_dim = self.num_continuous
            x_cont = x[:, :c_dim]
            x_cat = x[:, c_dim:]

            # Linear interpolation for continuous features
            x_cont_mix = lam * x_cont + (1.0 - lam) * x_cont[perm]

            # Categorical feature selection via Bernoulli mask
            cat_mask = (torch.rand_like(x_cat.float()) < lam).to(x_cat.dtype)
            x_cat_mix = cat_mask * x_cat + (1 - cat_mask) * x_cat[perm]

            x_mix = torch.cat([x_cont_mix, x_cat_mix], dim=1)
        else:
            # Full linear interpolation across all dimensions (standard for OHE tabular inputs)
            x_mix = lam * x + (1.0 - lam) * x[perm]

        # Mix ground-truth targets
        y_mix = lam * y + (1.0 - lam) * y[perm]

        # Mix teacher pseudo-label targets if provided
        y_teacher_mix = None
        if y_teacher is not None:
            y_teacher_mix = lam * y_teacher + (1.0 - lam) * y_teacher[perm]

        return x_mix, y_mix, y_teacher_mix, lam


# =============================================================================
# 4. FT-Transformer (Feature Tokenizer Transformer for Tabular Data)
# =============================================================================

class FeatureTokenizer(nn.Module):
    """
    Feature Tokenizer for Tabular Data (Gorishniy et al., NeurIPS 2021).
    
    Transforms each scalar continuous feature into a d-dimensional token vector:
      e_cont_j = x_j * W_j + b_j
    and each categorical feature via an entity embedding table:
      e_cat_k = Embedding_k(x_k)
      
    Prepends a learnable [CLS] token to the sequence of tokens.
    """

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: Optional[Sequence[int]] = None,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = list(cat_cardinalities) if cat_cardinalities else []
        self.embed_dim = embed_dim

        # Numerical feature projection: W is (num_continuous, embed_dim), b is (num_continuous, embed_dim)
        if num_continuous > 0:
            self.cont_weight = nn.Parameter(torch.empty(num_continuous, embed_dim))
            self.cont_bias = nn.Parameter(torch.empty(num_continuous, embed_dim))
            nn.init.kaiming_uniform_(self.cont_weight, a=math.sqrt(5))
            nn.init.zeros_(self.cont_bias)
        else:
            self.register_parameter("cont_weight", None)
            self.register_parameter("cont_bias", None)

        # Categorical entity embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card + 2, embedding_dim=embed_dim)
            for card in self.cat_cardinalities
        ])

        # Learnable [CLS] classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenizes tabular input matrix into a sequence of token vectors.
        
        Args:
            x: Input tensor of shape (B, D).
               If cat_cardinalities is present, x[:, :num_continuous] are continuous
               and x[:, num_continuous:] are categorical integer indices.
               If cat_cardinalities is empty, all columns are continuous.
               
        Returns:
            Token tensor of shape (B, 1 + num_features, embed_dim).
        """
        batch_size = x.size(0)
        tokens = []

        # 1. Continuous tokens
        if self.num_continuous > 0:
            x_cont = x[:, :self.num_continuous]
            # (B, num_cont, 1) * (1, num_cont, embed_dim) + (1, num_cont, embed_dim)
            cont_tokens = (
                x_cont.unsqueeze(-1) * self.cont_weight.unsqueeze(0)
                + self.cont_bias.unsqueeze(0)
            )
            tokens.append(cont_tokens)

        # 2. Categorical tokens
        if len(self.cat_embeddings) > 0:
            x_cat = x[:, self.num_continuous:].long()
            cat_token_list = []
            for i, emb in enumerate(self.cat_embeddings):
                c_idx = torch.clamp(x_cat[:, i], 0, emb.num_embeddings - 1)
                cat_token_list.append(emb(c_idx).unsqueeze(1))
            cat_tokens = torch.cat(cat_token_list, dim=1)
            tokens.append(cat_tokens)

        # Concatenate continuous + categorical tokens: (B, num_features, embed_dim)
        if tokens:
            feat_tokens = torch.cat(tokens, dim=1)
        else:
            feat_tokens = torch.empty((batch_size, 0, self.embed_dim), device=x.device)

        # Prepend [CLS] token: (B, 1 + num_features, embed_dim)
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)
        out = torch.cat([cls_expanded, feat_tokens], dim=1)
        return out


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Layer with Multihead Self-Attention and GELU MLP.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, ffn_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out
        # Pre-LN Feed-Forward with residual connection
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """
    Modern FT-Transformer (Feature Tokenizer Transformer) for Tabular Data.
    
    Implements the architecture from Gorishniy et al. (NeurIPS 2021) "Revisiting Deep
    Learning Models for Tabular Data".
    
    Args:
        num_continuous: Number of continuous/numerical features.
        cat_cardinalities: Optional list of cardinalities for categorical features.
                           If None or empty, all features are processed as continuous.
        embed_dim: Feature token embedding dimension (default 64).
        num_heads: Number of attention heads (default 4).
        num_layers: Number of transformer encoder layers (default 3).
        ffn_dim: Feed-forward network hidden dimension (default 128).
        dropout: Attention and FFN dropout rate (default 0.2).
        head_dropout: Classification head dropout rate (default 0.1).
    """

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: Optional[Sequence[int]] = None,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 128,
        dropout: float = 0.2,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = list(cat_cardinalities) if cat_cardinalities else []
        self.embed_dim = embed_dim

        # 1. Feature Tokenizer
        self.tokenizer = FeatureTokenizer(
            num_continuous=num_continuous,
            cat_cardinalities=self.cat_cardinalities,
            embed_dim=embed_dim,
        )

        # 2. Transformer Encoder Stack
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 3. Final Prediction Head (CLS token readout)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing binary stockout risk probability.
        
        Args:
            x: Input feature matrix of shape (B, D).
            
        Returns:
            Predicted probability tensor of shape (B, 1) in [0, 1].
        """
        # (B, 1 + M, embed_dim)
        h = self.tokenizer(x)

        for layer in self.layers:
            h = layer(h)

        # Extract [CLS] token representation at index 0: (B, embed_dim)
        cls_out = self.head_norm(h[:, 0])
        # (B, 1)
        prob = self.head(cls_out)
        return prob


# =============================================================================
# 5. Distillation Dataset Wrapper
# =============================================================================

class DistillationDataset(Dataset):
    """
    Dataset wrapper holding feature vectors, ground-truth labels, and teacher soft targets.
    """

    def __init__(
        self,
        features: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        teacher_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        if isinstance(features, np.ndarray):
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features.float()

        if isinstance(targets, np.ndarray):
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = targets.float()

        if self.targets.dim() == 1:
            self.targets = self.targets.unsqueeze(-1)

        if teacher_probs is not None:
            if isinstance(teacher_probs, np.ndarray):
                self.teacher_probs = torch.tensor(teacher_probs, dtype=torch.float32)
            else:
                self.teacher_probs = teacher_probs.float()
            if self.teacher_probs.dim() == 1:
                self.teacher_probs = self.teacher_probs.unsqueeze(-1)
        else:
            self.teacher_probs = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.features[idx]
        y = self.targets[idx]
        if self.teacher_probs is not None:
            p_t = self.teacher_probs[idx]
            return x, y, p_t
        return x, y, None


# =============================================================================
# 6. Tabular Knowledge Distillation Trainer
# =============================================================================

class TabularKnowledgeDistillationTrainer:
    """
    Unified Knowledge Distillation Engine for Tabular Deep Learning Models.
    
    Trains neural student models (such as FT-Transformer, TabResNet, HybridLSTMTransformer)
    using soft pseudo-label probability targets distilled from high-performing GBDT ensembles
    (LightGBM, XGBoost, CatBoost) combined with ground-truth Binary Focal Loss:
    
      Loss = alpha_distill * BCE(p_student, p_teacher) + (1 - alpha_distill) * FocalLoss(p_student, y_true)
      
    This effectively transfers the superior inductive bias of decision tree partitions to neural
    networks, bridging the historical performance gap between tree ensembles and deep models
    on tabular datasets.
    
    Features:
    - GPU acceleration with automatic AMP (fp16) on CUDA and seamless CPU fallback.
    - Full support for single GBDT models, multi-model ensemble averaging, or precomputed teacher arrays.
    - Integrated TabularMixup mini-batch augmentation.
    - Integrated CosineAnnealingWarmupRestarts learning rate schedule.
    - Validation PR-AUC early stopping and optimal F1 decision threshold tau* calibration.
    - Publication-grade telemetry and checkpoint management.
    
    Args:
        student_model: PyTorch neural network student (e.g. FTTransformer or TabResNet).
        model_name: Unique identifier string for checkpointing and logs.
        teacher_model: Optional trained GBDT model or list/ensemble of GBDT models with
                       `predict_proba` method. If None, teacher_probs must be supplied to fit().
        alpha_distill: Distillation loss balance factor in [0, 1]. Default 0.5.
                       (alpha_distill * DistillLoss + (1 - alpha_distill) * FocalLoss).
        temperature: Temperature scaling factor for soft probability calibration (default 1.0).
        focal_alpha: Minority class balance parameter for BinaryFocalLoss (default 0.25).
        focal_gamma: Focusing parameter for BinaryFocalLoss (default 2.0).
        scale_pos_weight: Optional positive-class weight for automatic focal_alpha derivation.
        lr: Base learning rate (default 1e-3).
        weight_decay: AdamW weight decay (default 1e-4).
        warmup_epochs: Initial linear warmup epochs (default 5).
        lr_scheduler_type: 'cosine_warmup', 'plateau', or 'none' (default 'cosine_warmup').
        use_mixup: Whether to activate TabularMixup mini-batch regularization (default False).
        mixup_alpha: Beta distribution parameter for mixup (default 0.2).
        mixup_prob: Probability of applying mixup to each mini-batch (default 0.5).
        gradient_clip: Gradient norm clipping maximum (default 1.0).
        use_amp: Whether to enable Automatic Mixed Precision on CUDA (default True).
        device: Target execution device. If None, automatically detected via DeviceManager.
        checkpoint_dir: Directory path for saving best checkpoint weights.
        seed: Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        student_model: nn.Module,
        model_name: str = "DistilledStudent",
        teacher_model: Optional[Any] = None,
        alpha_distill: float = 0.5,
        temperature: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        scale_pos_weight: Optional[float] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        lr_scheduler_type: str = "cosine_warmup",
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
        gradient_clip: float = 1.0,
        use_amp: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        max_epochs: int = 40,
        patience: int = 8,
        batch_size: int = 64,
        verbose: bool = True,
        require_gpu: bool = False,
    ):
        # Resolve device
        if device is None:
            self.device = _DEFAULT_DEVICE
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if require_gpu and self.device.type != "cuda":
            raise RuntimeError(f"--require-gpu failed loudly: GPU requested for {model_name} but resolved to {self.device}.")

        self.seed = seed
        torch.manual_seed(seed)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.student_model = student_model.to(self.device)
        self.model_name = model_name
        self.teacher_model = teacher_model
        self.alpha_distill = float(alpha_distill)
        self.temperature = float(temperature)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.lr_scheduler_type = lr_scheduler_type.lower()
        self.gradient_clip = float(gradient_clip)
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        self.default_max_epochs = int(max_epochs)
        self.default_patience = int(patience)
        self.default_batch_size = int(batch_size)
        self.default_verbose = bool(verbose)

        # Loss function setup
        self.focal_loss = BinaryFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            pos_weight=scale_pos_weight,
            reduction="mean",
        )
        self.eps = 1e-7

        # Mixup setup
        self.use_mixup = use_mixup
        self.mixup = TabularMixup(alpha=mixup_alpha, prob=mixup_prob)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir or MODELS_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / f"{model_name}_distilled_best.pt"

        # Optimizer and Scaler
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.scheduler: Optional[Any] = None

        # Tracking variables
        self.best_val_score = -float("inf")
        self.optimal_threshold = 0.5
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "distill_loss": [],
            "focal_loss": [],
            "val_loss": [],
            "val_pr_auc": [],
            "val_roc_auc": [],
            "val_f1": [],
            "learning_rate": [],
        }

    # -------------------------------------------------------------------------
    # Teacher Inference and Probability Extraction
    # -------------------------------------------------------------------------

    def extract_teacher_probs(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extracts soft probability predictions from the configured GBDT teacher or ensemble.
        
        Supports:
        - Single model with `predict_proba(X)`
        - Sequence/list of models (returns ensemble arithmetic average)
        - Torch tensor or numpy input
        """
        if self.teacher_model is None:
            raise ValueError(
                "teacher_model is None. Either provide teacher_model to trainer or pass "
                "teacher_probs directly into fit()."
            )

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if isinstance(self.teacher_model, (list, tuple)):
            # Ensemble average over multiple teachers (e.g. LGBM + XGBoost + CatBoost)
            prob_sum = np.zeros(X_np.shape[0], dtype=np.float32)
            for m in self.teacher_model:
                p = m.predict_proba(X_np)
                if p.ndim > 1 and p.shape[1] > 1:
                    prob_sum += p[:, 1]
                else:
                    prob_sum += p.ravel()
            teacher_probs = prob_sum / float(len(self.teacher_model))
        else:
            # Single teacher model
            p = self.teacher_model.predict_proba(X_np)
            if p.ndim > 1 and p.shape[1] > 1:
                teacher_probs = p[:, 1]
            else:
                teacher_probs = p.ravel()

        return teacher_probs.astype(np.float32)

    # -------------------------------------------------------------------------
    # Loss Formulation
    # -------------------------------------------------------------------------

    def compute_loss(
        self,
        p_student: torch.Tensor,
        y_true: torch.Tensor,
        p_teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Blended distillation loss computation:
          Loss = alpha_distill * DistillLoss + (1 - alpha_distill) * FocalLoss
          
        Args:
            p_student: Student predicted probabilities in [0, 1]. Shape (B, 1).
            y_true: Ground truth targets (binary or mixup interpolated). Shape (B, 1).
            p_teacher: Soft pseudo-label teacher predictions in [0, 1]. Shape (B, 1).
            
        Returns:
            Tuple of (total_loss, distill_loss_value, focal_loss_value).
        """
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(-1)

        # 1. Ground Truth Binary Focal Loss
        f_loss = self.focal_loss(p_student, y_true)

        # 2. Distillation Loss (Soft Cross-Entropy from Teacher)
        if p_teacher is not None and self.alpha_distill > 0.0:
            if p_teacher.dim() == 1:
                p_teacher = p_teacher.unsqueeze(-1)

            # Apply temperature scaling if T != 1.0
            if self.temperature != 1.0:
                p_s = torch.sigmoid(torch.logit(torch.clamp(p_student, self.eps, 1.0 - self.eps)) / self.temperature)
                p_t = torch.sigmoid(torch.logit(torch.clamp(p_teacher, self.eps, 1.0 - self.eps)) / self.temperature)
            else:
                p_s = torch.clamp(p_student, self.eps, 1.0 - self.eps)
                p_t = torch.clamp(p_teacher, self.eps, 1.0 - self.eps)

            # Soft Binary Cross Entropy between student and teacher
            d_loss = - torch.mean(
                p_t * torch.log(p_s) + (1.0 - p_t) * torch.log(1.0 - p_s)
            )
            # Scale by T^2 when temperature scaling is applied (Hinton et al.)
            if self.temperature != 1.0:
                d_loss = d_loss * (self.temperature ** 2)

            total_loss = (self.alpha_distill * d_loss) + ((1.0 - self.alpha_distill) * f_loss)
        else:
            d_loss = torch.tensor(0.0, device=p_student.device)
            total_loss = f_loss

        return total_loss, d_loss, f_loss

    # -------------------------------------------------------------------------
    # Training and Evaluation Epochs
    # -------------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Runs a single training epoch with Knowledge Distillation."""
        self.student_model.train()
        total_loss = 0.0
        total_distill = 0.0
        total_focal = 0.0
        total_samples = 0

        pin = (self.device.type == "cuda")

        for batch in dataloader:
            if len(batch) == 3:
                X_batch, y_batch, p_teacher_batch = batch
            else:
                X_batch, y_batch = batch
                p_teacher_batch = None

            X_batch = X_batch.to(self.device, non_blocking=pin)
            y_batch = y_batch.to(self.device, non_blocking=pin)
            if p_teacher_batch is not None:
                p_teacher_batch = p_teacher_batch.to(self.device, non_blocking=pin)

            # Optional Tabular Mixup
            if self.use_mixup and self.student_model.training:
                X_batch, y_batch, p_teacher_batch, _ = self.mixup(
                    X_batch, y_batch, p_teacher_batch
                )

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    p_student = self.student_model(X_batch)
                    loss, d_loss, f_loss = self.compute_loss(p_student, y_batch, p_teacher_batch)
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                p_student = self.student_model(X_batch)
                loss, d_loss, f_loss = self.compute_loss(p_student, y_batch, p_teacher_batch)
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=self.gradient_clip)
                self.optimizer.step()

            bs = len(X_batch)
            total_loss += loss.item() * bs
            total_distill += d_loss.item() * bs
            total_focal += f_loss.item() * bs
            total_samples += bs

        n = max(total_samples, 1)
        return total_loss / n, total_distill / n, total_focal / n

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """
        Evaluates student model on validation or test dataset.
        
        Returns:
            Tuple of (loss, roc_auc, pr_auc, f1_score_at_tau, predictions, ground_truth).
        """
        self.student_model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds: List[float] = []
        all_targets: List[float] = []

        pin = (self.device.type == "cuda")

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    X_batch, y_batch, p_teacher_batch = batch
                else:
                    X_batch, y_batch = batch
                    p_teacher_batch = None

                X_batch = X_batch.to(self.device, non_blocking=pin)
                y_batch = y_batch.to(self.device, non_blocking=pin)
                if p_teacher_batch is not None:
                    p_teacher_batch = p_teacher_batch.to(self.device, non_blocking=pin)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        p_student = self.student_model(X_batch)
                        loss, _, _ = self.compute_loss(p_student, y_batch, p_teacher_batch)
                else:
                    p_student = self.student_model(X_batch)
                    loss, _, _ = self.compute_loss(p_student, y_batch, p_teacher_batch)

                bs = len(X_batch)
                total_loss += loss.item() * bs
                total_samples += bs

                all_preds.extend(p_student.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        val_loss = total_loss / max(total_samples, 1)
        preds = np.array(all_preds, dtype=np.float32)
        targets = np.array(all_targets, dtype=np.float32)

        # Calculate ROC-AUC & PR-AUC
        roc = float(roc_auc_score(targets, preds)) if len(np.unique(targets)) > 1 else 0.5
        precisions, recalls, _ = precision_recall_curve(targets, preds)
        pr_auc = float(auc(recalls, precisions)) if len(recalls) > 1 else 0.0

        # Calculate F1 score at currently calibrated threshold
        bin_preds = (preds >= self.optimal_threshold).astype(int)
        f1 = float(f1_score(targets, bin_preds, zero_division=0))

        return val_loss, roc, pr_auc, f1, preds, targets

    def calibrate_threshold(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """Finds decision threshold tau* in [0.05, 0.95] maximizing F1 score."""
        best_thresh = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.05, 0.95, 91):
            bin_preds = (preds >= t).astype(int)
            f1 = f1_score(targets, bin_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        self.optimal_threshold = float(best_thresh)
        return self.optimal_threshold

    # -------------------------------------------------------------------------
    # Master Fit Routine
    # -------------------------------------------------------------------------

    def fit(
        self,
        train_data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]],
        val_data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]],
        teacher_probs_train: Optional[np.ndarray] = None,
        teacher_probs_val: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Knowledge Distillation training loop with early stopping on PR-AUC.
        
        Args:
            train_data: DataLoader or tuple (X_train, y_train).
            val_data: DataLoader or tuple (X_val, y_val).
            teacher_probs_train: Optional precomputed soft teacher probabilities for train set.
                                 If None and teacher_model was passed in __init__, auto-computed.
            teacher_probs_val: Optional precomputed soft teacher probabilities for val set.
            batch_size: Batch size when constructing loaders from numpy arrays (defaults to init).
            max_epochs: Maximum epochs (defaults to init).
            patience: Early stopping patience epochs on validation PR-AUC (defaults to init).
            verbose: Print epoch-by-epoch progress (defaults to init).
            
        Returns:
            Dictionary containing best validation metrics, optimal threshold, and checkpoint path.
        """
        b_size = batch_size if batch_size is not None else self.default_batch_size
        epochs = max_epochs if max_epochs is not None else self.default_max_epochs
        pat = patience if patience is not None else self.default_patience
        verb = verbose if verbose is not None else self.default_verbose
        pin = (self.device.type == "cuda")

        # 1. Prepare Train DataLoader with soft targets
        if isinstance(train_data, tuple):
            X_tr, y_tr = train_data
            if teacher_probs_train is None and self.teacher_model is not None:
                if verb:
                    print(f"[{self.model_name}] Distilling soft pseudo-labels from GBDT teacher on train set...")
                teacher_probs_train = self.extract_teacher_probs(X_tr)

            train_ds = DistillationDataset(X_tr, y_tr, teacher_probs_train)
            train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True, pin_memory=pin)
        else:
            train_loader = train_data

        # 2. Prepare Validation DataLoader
        if isinstance(val_data, tuple):
            X_va, y_va = val_data
            if teacher_probs_val is None and self.teacher_model is not None:
                teacher_probs_val = self.extract_teacher_probs(X_va)

            val_ds = DistillationDataset(X_va, y_va, teacher_probs_val)
            val_loader = DataLoader(val_ds, batch_size=b_size, shuffle=False, pin_memory=pin)
        else:
            val_loader = val_data

        # 3. Configure Learning Rate Scheduler
        if self.lr_scheduler_type == "cosine_warmup":
            w_steps = min(self.warmup_epochs, max(0, epochs - 1))
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=epochs,
                cycle_mult=1.0,
                max_lr=self.lr,
                min_lr=1e-6,
                warmup_steps=w_steps,
            )
        elif self.lr_scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=3
            )
        else:
            self.scheduler = None

        patience_counter = 0

        if verb:
            print("=" * 80)
            print(f"STARTING TABULAR KNOWLEDGE DISTILLATION: Student={self.model_name}")
            print(f"Alpha Distill: {self.alpha_distill:.2f} | Temperature: {self.temperature:.1f} | "
                  f"Device: {self.device} | Mixup: {self.use_mixup}")
            print("=" * 80)

        for epoch in range(1, epochs + 1):
            t_loss, d_loss, f_loss = self.train_epoch(train_loader)
            v_loss, v_roc, v_pr_auc, v_f1, val_preds, val_targets = self.evaluate(val_loader)

            curr_lr = self.optimizer.param_groups[0]["lr"]
            self.training_history["train_loss"].append(t_loss)
            self.training_history["distill_loss"].append(d_loss)
            self.training_history["focal_loss"].append(f_loss)
            self.training_history["val_loss"].append(v_loss)
            self.training_history["val_pr_auc"].append(v_pr_auc)
            self.training_history["val_roc_auc"].append(v_roc)
            self.training_history["val_f1"].append(v_f1)
            self.training_history["learning_rate"].append(curr_lr)

            # Step scheduler
            if isinstance(self.scheduler, CosineAnnealingWarmupRestarts):
                self.scheduler.step()
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(v_pr_auc)

            # Early stopping check on validation PR-AUC
            if v_pr_auc > self.best_val_score:
                self.best_val_score = v_pr_auc
                self.calibrate_threshold(val_preds, val_targets)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.student_model.state_dict(),
                        "optimal_threshold": self.optimal_threshold,
                        "best_val_pr_auc": self.best_val_score,
                        "val_roc_auc": v_roc,
                        "seed": self.seed,
                        "model_name": self.model_name,
                        "alpha_distill": self.alpha_distill,
                    },
                    self.checkpoint_path,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if verb and (epoch % 5 == 0 or patience_counter == 0 or epoch == epochs):
                print(
                    f"[{self.model_name}|Seed {self.seed}] Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {t_loss:.4f} (D: {d_loss:.4f}, F: {f_loss:.4f}) | "
                    f"Val Loss: {v_loss:.4f} | PR-AUC: {v_pr_auc:.4f} | ROC: {v_roc:.4f} | "
                    f"tau*: {self.optimal_threshold:.2f} | LR: {curr_lr:.2e}"
                )

            if patience_counter >= pat:
                if verb:
                    print(f"[{self.model_name}|Seed {self.seed}] Early stopping triggered at epoch {epoch}.")
                break

        # Load best weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.student_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimal_threshold = float(checkpoint.get("optimal_threshold", 0.5))

        if verb:
            print(f"[+] Distillation complete! Best Val PR-AUC: {self.best_val_score:.4f} "
                  f"with optimal tau*: {self.optimal_threshold:.2f}")

        return {
            "best_val_pr_auc": self.best_val_score,
            "optimal_threshold": self.optimal_threshold,
            "checkpoint_path": str(self.checkpoint_path),
            "history": self.training_history,
        }

    # -------------------------------------------------------------------------
    # Inference Utilities
    # -------------------------------------------------------------------------

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], batch_size: int = 128) -> np.ndarray:
        """
        Generates probability predictions on new samples.
        
        Args:
            X: Input feature array or tensor of shape (N, D).
            batch_size: Batch size for inference.
            
        Returns:
            Numpy array of predicted delay probabilities of shape (N,).
        """
        self.student_model.eval()
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.float()

        dataset = TensorDataset(X_tensor)
        pin = (self.device.type == "cuda")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)

        probs: List[float] = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device, non_blocking=pin)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        out = self.student_model(bx)
                else:
                    out = self.student_model(bx)
                probs.extend(out.cpu().numpy().flatten())

        return np.array(probs, dtype=np.float32)

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        threshold: Optional[float] = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Generates discrete binary predictions using calibrated optimal threshold tau*.
        """
        tau = self.optimal_threshold if threshold is None else float(threshold)
        probs = self.predict_proba(X, batch_size=batch_size)
        return (probs >= tau).astype(int)


__all__ = [
    "BinaryFocalLoss",
    "CosineAnnealingWarmupRestarts",
    "TabularMixup",
    "FeatureTokenizer",
    "TransformerBlock",
    "FTTransformer",
    "DistillationDataset",
    "TabularKnowledgeDistillationTrainer",
]
