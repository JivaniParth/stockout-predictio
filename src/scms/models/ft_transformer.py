"""
FT-Transformer (Feature Tokenizer Transformer) and Attentive TabResNet (TabResNet-Plus)
for Pharmaceutical Delivery Delay and Stockout Risk Prediction (USAID SCMS).

Key Components:
1. NumericalFeatureTokenizer: Vectorized linear tokenization projecting each continuous
   numerical scalar into a d-dimensional token vector (d=64).
2. CategoricalFeatureTokenizer: Entity embeddings projecting categorical columns into
   d-dimensional token vectors with out-of-vocabulary protection.
3. Learnable [CLS] Classification Token prepended to the feature token sequence.
4. Transformer Encoder: 2-3 Pre-LayerNorm encoder blocks with MultiheadAttention (4-8 heads),
   configurable SwiGLU / GeLU feedforward networks, dropout (0.15-0.20), and residual connections.
5. Classifier Head: Linear projection from [CLS] token to binary delivery delay probability.
6. FTTransformerWrapper: Scikit-learn compatible wrapper providing fit(X_train, y_train, X_val, y_val),
   predict_proba(X), early stopping on validation PR-AUC, threshold calibration, and GPU acceleration
   (device='cuda') with automatic graceful fallback to multi-core CPU parallel execution.
7. AttentiveTabResNet & AttentiveTabResNetWrapper: Augments tabular residual dense blocks
   with multi-head self-attention feature interaction layers and multi-scale skip connections.
"""

import os
import math
import copy
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score

try:
    from src.config import DEVICE, RANDOM_SEED
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Decision Threshold Utility
# -----------------------------------------------------------------------------

def calibrate_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_steps: int = 81,
    min_thresh: float = 0.1,
    max_thresh: float = 0.9,
) -> float:
    """
    Calibrates decision threshold tau* maximizing F1 score on validation delay occurrences.
    Useful for addressing positive class imbalance in pharmaceutical delivery delay prediction.
    """
    thresholds = np.linspace(min_thresh, max_thresh, n_steps)
    best_thresh = 0.5
    best_f1 = -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    return best_thresh


# -----------------------------------------------------------------------------
# 1. Feature Tokenizers
# -----------------------------------------------------------------------------

class NumericalFeatureTokenizer(nn.Module):
    """
    Vectorized Linear Feature Tokenizer for continuous numerical variables.
    Projects each numerical scalar x_j into a d-dimensional token vector:
        T_j = x_j * W_j + b_j
    where W_j in R^d and b_j in R^d are trainable parameters.

    Reference: Gorishniy et al., 'Revisiting Deep Learning Models for Tabular Data' (NeurIPS 2021).
    """

    def __init__(self, num_features: int, d_token: int = 64, bias: bool = True):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token

        if num_features > 0:
            self.weight = nn.Parameter(torch.empty(num_features, d_token))
            if bias:
                self.bias = nn.Parameter(torch.empty(num_features, d_token))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = 1
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Continuous features tensor of shape (batch_size, num_features).
        Returns:
            Token tensor of shape (batch_size, num_features, d_token).
        """
        if self.num_features == 0 or self.weight is None:
            return torch.empty((x.size(0), 0, self.d_token), device=x.device, dtype=x.dtype)

        # Vectorized projection across all numerical features:
        # x.unsqueeze(-1): (B, num_features, 1)
        # self.weight.unsqueeze(0): (1, num_features, d_token)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0)
        if self.bias is not None:
            tokens = tokens + self.bias.unsqueeze(0)
        return tokens


class CategoricalFeatureTokenizer(nn.Module):
    """
    Learned Entity Embedding Tokenizer for categorical tabular variables.
    Projects each categorical column k into a d-dimensional token vector:
        T_k = Embedding_k(x_k)
    Includes out-of-vocabulary protection and categorical index clamping.
    """

    def __init__(self, cat_cardinalities: List[int], d_token: int = 64):
        super().__init__()
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.d_token = d_token

        # Allocate num_embeddings = card + 2 for unseen/padding index safety
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=max(int(card), 1) + 2, embedding_dim=d_token)
            for card in self.cat_cardinalities
        ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Categorical features tensor of shape (batch_size, num_categoricals).
        Returns:
            Token tensor of shape (batch_size, num_categoricals, d_token).
        """
        if len(self.embeddings) == 0:
            return torch.empty((x.size(0), 0, self.d_token), device=x.device, dtype=torch.float32)

        tokens = []
        for i, emb in enumerate(self.embeddings):
            # Clamp index to [0, num_embeddings - 1] to prevent out-of-bounds indexing
            idx = torch.clamp(x[:, i].long(), 0, emb.num_embeddings - 1)
            tokens.append(emb(idx).unsqueeze(1))  # (B, 1, d_token)

        return torch.cat(tokens, dim=1)  # (B, num_categoricals, d_token)


# -----------------------------------------------------------------------------
# 2. Feedforward Networks: SwiGLU & GeLU
# -----------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network.
    Paper: Shazeer, 'GLU Variants Improve Transformer' (2020).
    SwiGLU(x) = (SiLU(x W_gate) * (x W_val)) W_out
    """

    def __init__(self, d_model: int, d_ffn: int, dropout_rate: float = 0.2):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.val_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.out_proj = nn.Linear(d_ffn, d_model, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        val = self.val_proj(x)
        h = self.dropout(gate * val)
        return self.out_proj(h)


class GeLUFeedForward(nn.Module):
    """
    Standard Position-wise Feed-Forward Network with GELU non-linearity.
    FFN(x) = Dropout(W_2 * Dropout(GELU(W_1 * x + b_1)) + b_2)
    """

    def __init__(self, d_model: int, d_ffn: int, dropout_rate: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout1(F.gelu(self.linear1(x)))
        return self.dropout2(self.linear2(h))


# -----------------------------------------------------------------------------
# 3. Transformer Encoder Block (Pre-LayerNorm)
# -----------------------------------------------------------------------------

class FTTransformerEncoderBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Block.
    Applies Multihead Self-Attention followed by Position-Wise Feedforward (SwiGLU or GeLU),
    each surrounded by Pre-LayerNorm and residual skip connections.
    """

    def __init__(
        self,
        d_token: int = 64,
        num_heads: int = 8,
        ffn_dim: int = 128,
        ffn_type: str = "swiglu",
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout_mha = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(d_token)
        if ffn_type.lower() == "swiglu":
            self.ffn = SwiGLU(d_model=d_token, d_ffn=ffn_dim, dropout_rate=dropout_rate)
        else:
            self.ffn = GeLUFeedForward(d_model=d_token, d_ffn=ffn_dim, dropout_rate=dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm Self-Attention + Residual
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(query=norm_x, key=norm_x, value=norm_x)
        x = x + self.dropout_mha(attn_out)

        # Pre-LayerNorm Feed-Forward + Residual
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout_ffn(ffn_out)
        return x


# -----------------------------------------------------------------------------
# 4. FT-Transformer (Feature Tokenizer Transformer)
# -----------------------------------------------------------------------------

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer (FT-Transformer) for Tabular Data.
    Adapted for Pharmaceutical Supply Chain Delivery Delay / Stockout Risk Prediction.

    Architecture:
    1. Numerical Feature Tokenizer: Linear projection of continuous features -> (B, n_num, d)
    2. Categorical Feature Tokenizer: Learned entity embeddings -> (B, n_cat, d)
    3. Learnable [CLS] Classification Token prepended to sequence -> (B, 1 + n_feat, d)
    4. Stack of Pre-LayerNorm Transformer Encoder Blocks with Multihead Self-Attention
    5. Final LayerNorm + Linear Classifier Head on [CLS] token -> binary delay probability
    """

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: Optional[List[int]] = None,
        d_token: int = 64,
        num_blocks: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 128,
        ffn_type: str = "swiglu",
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = list(cat_cardinalities) if cat_cardinalities is not None else []
        self.d_token = d_token
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.ffn_type = ffn_type

        # 1. Feature Tokenizers
        self.num_tokenizer = NumericalFeatureTokenizer(num_features=num_continuous, d_token=d_token)
        self.cat_tokenizer = CategoricalFeatureTokenizer(cat_cardinalities=self.cat_cardinalities, d_token=d_token)

        # 2. [CLS] Classification Token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # 3. Transformer Encoder Stack
        self.blocks = nn.ModuleList([
            FTTransformerEncoderBlock(
                d_token=d_token,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                ffn_type=ffn_type,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_blocks)
        ])

        # 4. Final Layer Normalization
        self.final_norm = nn.LayerNorm(d_token)

        # 5. Classifier Head: Linear projection from [CLS] token to binary delay probability
        self.head = nn.Linear(d_token, 1)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of FT-Transformer.

        Args:
            x: Either a single tensor (batch_size, num_continuous + num_categorical)
               or a tuple of (x_continuous, x_categorical).
            return_logits: If True, returns unbounded logits. If False, returns Sigmoid probabilities.
        Returns:
            Tensor of shape (batch_size, 1) with delay probabilities or logits.
        """
        if isinstance(x, (tuple, list)):
            x_cont, x_cat = x[0], x[1]
            batch_size = x_cont.size(0) if x_cont is not None else x_cat.size(0)
            device = x_cont.device if x_cont is not None else x_cat.device
        else:
            batch_size = x.size(0)
            device = x.device
            x_cont = x[:, : self.num_continuous] if self.num_continuous > 0 else None
            x_cat = x[:, self.num_continuous :] if len(self.cat_cardinalities) > 0 else None

        tokens_list = []
        if self.num_continuous > 0 and x_cont is not None:
            tokens_list.append(self.num_tokenizer(x_cont.float()))
        if len(self.cat_cardinalities) > 0 and x_cat is not None:
            tokens_list.append(self.cat_tokenizer(x_cat.long()))

        if tokens_list:
            feat_tokens = torch.cat(tokens_list, dim=1)  # (B, n_features, d_token)
        else:
            feat_tokens = torch.empty((batch_size, 0, self.d_token), device=device, dtype=torch.float32)

        # Prepend learnable [CLS] token
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_token)
        h = torch.cat([cls_expanded, feat_tokens], dim=1)  # (B, 1 + n_features, d_token)

        # Pass through Transformer encoder blocks
        for block in self.blocks:
            h = block(h)

        # Extract [CLS] token representation
        cls_out = self.final_norm(h[:, 0, :])  # (B, d_token)

        logits = self.head(cls_out)  # (B, 1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


# -----------------------------------------------------------------------------
# 5. Attentive TabResNet (TabResNet-Plus)
# -----------------------------------------------------------------------------

class TabularResBlock(nn.Module):
    """
    Residual dense block with normalization, non-linear activation, and dropout.
    Supports both LayerNorm and BatchNorm1d for robust single-sample inference.
    """

    def __init__(self, dim: int, dropout_rate: float = 0.2, use_layernorm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) if use_layernorm else nn.BatchNorm1d(dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim) if use_layernorm else nn.BatchNorm1d(dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.linear1(self.drop1(self.act1(self.norm1(x))))
        h = self.linear2(self.drop2(self.act2(self.norm2(h))))
        return residual + h


class AttentiveTabResNet(nn.Module):
    """
    Attentive TabResNet (TabResNet-Plus) Architecture.
    Combines feature tokenization, multi-head self-attention feature interaction layers,
    attentive pooling, dense residual blocks, and multi-scale skip connections.

    Key Stages:
    1. Feature Tokenization: Continuous (linear) and Categorical (entity embedding) into d-dim tokens.
    2. Multi-Head Self-Attention Interaction: Feature tokens attend to each other to capture
       high-order pairwise and group feature interactions.
    3. Attentive Pooling & Compression: Feature-wise attentive context aggregation.
    4. Residual Dense Blocks: Stack of deep residual blocks with skip connections.
    5. Classifier Head: Output projection to delay probability.
    """

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: Optional[List[int]] = None,
        d_token: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = list(cat_cardinalities) if cat_cardinalities is not None else []
        self.d_token = d_token
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_features = num_continuous + len(self.cat_cardinalities)

        # 1. Feature Tokenizers
        self.num_tokenizer = NumericalFeatureTokenizer(num_features=num_continuous, d_token=d_token)
        self.cat_tokenizer = CategoricalFeatureTokenizer(cat_cardinalities=self.cat_cardinalities, d_token=d_token)

        # 2. Multi-Head Self-Attention Feature Interaction Layer
        self.attn_norm = nn.LayerNorm(d_token)
        self.feature_mha = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout_rate)

        # 3. Attentive Pooling
        self.pool_scorer = nn.Sequential(
            nn.Linear(d_token, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # 4. Dense Input Fusion
        # Fuses attentive pooled representation (d_token) + feature-wise representation
        in_dim = d_token + (self.num_features * d_token) if self.num_features > 0 else d_token
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 5. Residual Dense Blocks Stack
        self.res_blocks = nn.ModuleList([
            TabularResBlock(dim=hidden_dim, dropout_rate=dropout_rate, use_layernorm=True)
            for _ in range(num_res_blocks)
        ])

        # 6. Classifier Head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Single tensor (batch_size, num_continuous + num_categorical) or tuple of (x_cont, x_cat).
            return_logits: If True, returns raw logits. If False, returns Sigmoid probabilities.
        Returns:
            Tensor of shape (batch_size, 1) with delay probabilities or logits.
        """
        if isinstance(x, (tuple, list)):
            x_cont, x_cat = x[0], x[1]
            batch_size = x_cont.size(0) if x_cont is not None else x_cat.size(0)
            device = x_cont.device if x_cont is not None else x_cat.device
        else:
            batch_size = x.size(0)
            device = x.device
            x_cont = x[:, : self.num_continuous] if self.num_continuous > 0 else None
            x_cat = x[:, self.num_continuous :] if len(self.cat_cardinalities) > 0 else None

        tokens_list = []
        if self.num_continuous > 0 and x_cont is not None:
            tokens_list.append(self.num_tokenizer(x_cont.float()))
        if len(self.cat_cardinalities) > 0 and x_cat is not None:
            tokens_list.append(self.cat_tokenizer(x_cat.long()))

        if tokens_list:
            feat_tokens = torch.cat(tokens_list, dim=1)  # (B, N_feat, d_token)
        else:
            feat_tokens = torch.empty((batch_size, 0, self.d_token), device=device, dtype=torch.float32)

        if self.num_features > 0:
            # Multi-Head Self-Attention interaction
            norm_tokens = self.attn_norm(feat_tokens)
            attn_out, _ = self.feature_mha(query=norm_tokens, key=norm_tokens, value=norm_tokens)
            inter_tokens = feat_tokens + self.attn_dropout(attn_out)  # (B, N_feat, d_token)

            # Attentive Pooling over all feature tokens
            pool_weights = F.softmax(self.pool_scorer(inter_tokens), dim=1)  # (B, N_feat, 1)
            pooled_context = torch.sum(inter_tokens * pool_weights, dim=1)   # (B, d_token)

            # Flattened token interaction representation
            flat_tokens = inter_tokens.reshape(batch_size, -1)               # (B, N_feat * d_token)
            fused = torch.cat([pooled_context, flat_tokens], dim=1)          # (B, in_dim)
        else:
            fused = torch.zeros((batch_size, self.d_token), device=device, dtype=torch.float32)

        # Dense projection into residual space
        h = self.in_proj(fused)

        # Deep residual processing
        for block in self.res_blocks:
            h = block(h)

        logits = self.head(h)  # (B, 1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


# -----------------------------------------------------------------------------
# 6. FT-Transformer Wrapper (Scikit-Learn Interface)
# -----------------------------------------------------------------------------

class FTTransformerWrapper:
    """
    Production Scikit-Learn Compatible Wrapper for FT-Transformer.

    Features:
    - GPU Acceleration (CUDA pinned memory, non-blocking asynchronous streaming)
    - Automatic Graceful Fallback to multi-core parallel CPU execution on CUDA errors/OOM
    - Validation PR-AUC early stopping checkpointing (optimal model weight restoration)
    - Decision threshold calibration maximizing validation delay F1 score
    - Full telemetry tracking (active_device, fallback_occurred, training_history)
    - save() and load() persistence methods
    """

    def __init__(
        self,
        num_continuous: Optional[int] = None,
        cat_cardinalities: Optional[List[int]] = None,
        d_token: int = 64,
        num_blocks: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 128,
        ffn_type: str = "swiglu",
        dropout_rate: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scale_pos_weight: float = 6.0,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = RANDOM_SEED,
    ):
        # Resolve initial execution device with proactive CUDA availability check
        if device is None:
            self.device = DEVICE if isinstance(DEVICE, torch.device) else torch.device(DEVICE)
        elif isinstance(device, str):
            if device.lower() in ["cuda", "gpu"] and not torch.cuda.is_available():
                logger.warning(
                    f"[FTTransformerWrapper] CUDA requested ('{device}') but CUDA is not available. "
                    "Proactively falling back to multi-core CPU parallel execution."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        elif isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "[FTTransformerWrapper] CUDA device requested but CUDA is not available. "
                    "Proactively falling back to multi-core CPU parallel execution."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                self.device = torch.device("cpu")
            else:
                self.device = device
        else:
            self.device = torch.device("cpu")

        if not hasattr(self, "fallback_occurred") or not self.fallback_occurred:
            self.active_device = self.device.type
            self.fallback_occurred = False

        self.num_continuous = num_continuous
        self.cat_cardinalities = list(cat_cardinalities) if cat_cardinalities is not None else []
        self.d_token = d_token
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.ffn_type = ffn_type
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.scale_pos_weight = scale_pos_weight

        self.optimal_threshold: float = 0.5
        self.best_val_pr_auc: float = -1.0
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_pr_auc": [],
            "val_roc_auc": [],
            "learning_rate": [],
        }

        self.model: Optional[FTTransformer] = None
        if num_continuous is not None:
            self._build_model()

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _build_model(self) -> None:
        assert self.num_continuous is not None, "num_continuous must be set before building model"
        self.model = FTTransformer(
            num_continuous=self.num_continuous,
            cat_cardinalities=self.cat_cardinalities,
            d_token=self.d_token,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            ffn_type=self.ffn_type,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

    def _sanitize_array(self, X: Any) -> np.ndarray:
        if hasattr(X, "values"):
            arr = X.values
        elif isinstance(X, torch.Tensor):
            arr = X.detach().cpu().numpy()
        else:
            arr = np.asarray(X)
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def _fit_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        max_epochs: int,
        batch_size: int,
        patience: int,
    ) -> "FTTransformerWrapper":
        assert self.model is not None

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=max(1, patience // 2)
        )
        pos_w = torch.tensor([self.scale_pos_weight], device=self.device)

        pin = (self.device.type == "cuda")
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        drop_last = (len(train_ds) > batch_size and len(train_ds) % batch_size == 1)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin,
            drop_last=drop_last,
        )

        best_pr_auc = -1.0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_ctr = 0

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for bx, by in train_loader:
                bx = bx.to(self.device, non_blocking=pin)
                by = by.to(self.device, non_blocking=pin)

                optimizer.zero_grad()
                logits = self.model(bx, return_logits=True).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, by, pos_weight=pos_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_train_loss = epoch_loss / max(1, n_batches)
            current_lr = optimizer.param_groups[0]["lr"]

            # Validation evaluation
            val_pr_auc = 0.0
            val_roc_auc = 0.5
            val_loss = 0.0

            if X_val is not None and y_val is not None:
                val_probs = self._predict_proba_loop(X_val, batch_size=batch_size * 2)[:, 1]
                if len(np.unique(y_val)) > 1:
                    p, r, _ = precision_recall_curve(y_val, val_probs)
                    val_pr_auc = float(auc(r, p))
                    try:
                        val_roc_auc = float(roc_auc_score(y_val, val_probs))
                    except Exception:
                        val_roc_auc = 0.5
                else:
                    val_pr_auc = 0.0

                scheduler.step(val_pr_auc)

                if val_pr_auc > best_pr_auc:
                    best_pr_auc = val_pr_auc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        logger.info(f"[FTTransformerWrapper] Early stopping triggered at epoch {epoch + 1}")
                        break

            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_pr_auc"].append(val_pr_auc)
            self.training_history["val_roc_auc"].append(val_roc_auc)
            self.training_history["learning_rate"].append(current_lr)

        # Restore best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_val_pr_auc = best_pr_auc

        # Calibrate decision threshold on validation set
        if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            val_probs = self._predict_proba_loop(X_val, batch_size=batch_size * 2)[:, 1]
            self.optimal_threshold = calibrate_decision_threshold(y_val, val_probs)
        else:
            self.optimal_threshold = 0.5

        return self

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        max_epochs: int = 30,
        batch_size: int = 128,
        patience: int = 7,
    ) -> "FTTransformerWrapper":
        """
        Trains the FT-Transformer with validation PR-AUC early stopping.
        Automatically catches CUDA errors/OOM and falls back to multi-core CPU parallel execution.
        """
        X_train_arr = self._sanitize_array(X_train)
        y_train_arr = np.asarray(y_train, dtype=np.float32).ravel()

        X_val_arr = self._sanitize_array(X_val) if X_val is not None else None
        y_val_arr = np.asarray(y_val, dtype=np.float32).ravel() if y_val is not None else None

        # Automatically infer dimensions if not set
        if self.model is None or self.num_continuous is None:
            total_cols = X_train_arr.shape[1]
            cat_cols = len(self.cat_cardinalities)
            self.num_continuous = max(0, total_cols - cat_cols)
            self._build_model()

        try:
            return self._fit_loop(X_train_arr, y_train_arr, X_val_arr, y_val_arr, max_epochs, batch_size, patience)
        except Exception as e:
            if self.device.type == "cuda":
                logger.warning(
                    f"[FTTransformerWrapper] CUDA training encountered: {e}. "
                    "Gracefully falling back to multi-core CPU parallel processing..."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                try:
                    torch.set_num_threads(os.cpu_count() or 4)
                except Exception:
                    pass
                assert self.model is not None
                self.model = self.model.to(self.device)
                return self._fit_loop(X_train_arr, y_train_arr, X_val_arr, y_val_arr, max_epochs, batch_size, patience)
            else:
                raise e

    def _predict_proba_loop(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        all_p1 = []

        with torch.no_grad():
            N = len(X)
            for i in range(0, N, batch_size):
                chunk = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=self.device)
                p1_chunk = self.model(chunk, return_logits=False).squeeze(-1).cpu().numpy().flatten()
                all_p1.append(p1_chunk)

        p1 = np.concatenate(all_p1) if all_p1 else np.empty(0, dtype=np.float32)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict_proba(self, X: Any, batch_size: int = 1024) -> np.ndarray:
        """
        Computes binary probability distribution [P(On-time), P(Late Delivery Risk)].

        Args:
            X: Input tabular features matrix of shape (N, num_features).
            batch_size: Inference evaluation batch size.
        Returns:
            Numpy array of shape (N, 2).
        """
        X_arr = self._sanitize_array(X)
        try:
            return self._predict_proba_loop(X_arr, batch_size=batch_size)
        except Exception as e:
            if self.device.type == "cuda":
                logger.warning(
                    f"[FTTransformerWrapper] CUDA inference failed with error: {e}. "
                    "Falling back to multi-core CPU inference..."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                assert self.model is not None
                self.model = self.model.to(self.device)
                return self._predict_proba_loop(X_arr, batch_size=batch_size)
            else:
                raise e

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        """
        Binary prediction using calibrated decision threshold.
        """
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thresh).astype(int)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves model weights and wrapper metadata to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        payload = {
            "model_state": self.model.state_dict(),
            "num_continuous": self.num_continuous,
            "cat_cardinalities": self.cat_cardinalities,
            "d_token": self.d_token,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "ffn_type": self.ffn_type,
            "dropout_rate": self.dropout_rate,
            "optimal_threshold": self.optimal_threshold,
            "best_val_pr_auc": self.best_val_pr_auc,
            "training_history": self.training_history,
        }
        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[Union[str, torch.device]] = None) -> "FTTransformerWrapper":
        """Loads fitted FTTransformerWrapper checkpoint from disk."""
        filepath = Path(filepath)
        payload = torch.load(filepath, map_location="cpu")
        wrapper = cls(
            num_continuous=payload["num_continuous"],
            cat_cardinalities=payload["cat_cardinalities"],
            d_token=payload["d_token"],
            num_blocks=payload["num_blocks"],
            num_heads=payload["num_heads"],
            ffn_dim=payload["ffn_dim"],
            ffn_type=payload["ffn_type"],
            dropout_rate=payload["dropout_rate"],
            device=device,
        )
        assert wrapper.model is not None
        wrapper.model.load_state_dict(payload["model_state"])
        wrapper.optimal_threshold = payload.get("optimal_threshold", 0.5)
        wrapper.best_val_pr_auc = payload.get("best_val_pr_auc", -1.0)
        wrapper.training_history = payload.get("training_history", {})
        return wrapper


# -----------------------------------------------------------------------------
# 7. Attentive TabResNet Wrapper (Scikit-Learn Interface)
# -----------------------------------------------------------------------------

class AttentiveTabResNetWrapper:
    """
    Production Scikit-Learn Compatible Wrapper for Attentive TabResNet (TabResNet-Plus).

    Features:
    - Multi-head self-attention interaction augmented with deep tabular residual blocks
    - CUDA GPU acceleration with non-blocking streaming
    - Graceful CPU parallel fallback on CUDA error or OOM
    - Validation PR-AUC early stopping and optimal threshold calibration
    - Telemetry tracking and persistence
    """

    def __init__(
        self,
        num_continuous: Optional[int] = None,
        cat_cardinalities: Optional[List[int]] = None,
        d_token: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout_rate: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scale_pos_weight: float = 6.0,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = RANDOM_SEED,
    ):
        # Resolve initial execution device with proactive CUDA availability check
        if device is None:
            self.device = DEVICE if isinstance(DEVICE, torch.device) else torch.device(DEVICE)
        elif isinstance(device, str):
            if device.lower() in ["cuda", "gpu"] and not torch.cuda.is_available():
                logger.warning(
                    f"[AttentiveTabResNetWrapper] CUDA requested ('{device}') but CUDA is not available. "
                    "Proactively falling back to multi-core CPU parallel execution."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        elif isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "[AttentiveTabResNetWrapper] CUDA device requested but CUDA is not available. "
                    "Proactively falling back to multi-core CPU parallel execution."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                self.device = torch.device("cpu")
            else:
                self.device = device
        else:
            self.device = torch.device("cpu")

        self.seed = seed
        self._set_seed(seed)

        if not hasattr(self, "fallback_occurred") or not self.fallback_occurred:
            self.active_device = self.device.type
            self.fallback_occurred = False

        self.num_continuous = num_continuous
        self.cat_cardinalities = list(cat_cardinalities) if cat_cardinalities is not None else []
        self.d_token = d_token
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.scale_pos_weight = scale_pos_weight

        self.optimal_threshold: float = 0.5
        self.best_val_pr_auc: float = -1.0
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_pr_auc": [],
            "val_roc_auc": [],
            "learning_rate": [],
        }

        self.model: Optional[AttentiveTabResNet] = None
        if num_continuous is not None:
            self._build_model()

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _build_model(self) -> None:
        assert self.num_continuous is not None, "num_continuous must be set before building model"
        self.model = AttentiveTabResNet(
            num_continuous=self.num_continuous,
            cat_cardinalities=self.cat_cardinalities,
            d_token=self.d_token,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            num_res_blocks=self.num_res_blocks,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

    def _sanitize_array(self, X: Any) -> np.ndarray:
        if hasattr(X, "values"):
            arr = X.values
        elif isinstance(X, torch.Tensor):
            arr = X.detach().cpu().numpy()
        else:
            arr = np.asarray(X)
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def _fit_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        max_epochs: int,
        batch_size: int,
        patience: int,
    ) -> "AttentiveTabResNetWrapper":
        assert self.model is not None

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=max(1, patience // 2)
        )
        pos_w = torch.tensor([self.scale_pos_weight], device=self.device)

        pin = (self.device.type == "cuda")
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        drop_last = (len(train_ds) > batch_size and len(train_ds) % batch_size == 1)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin,
            drop_last=drop_last,
        )

        best_pr_auc = -1.0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_ctr = 0

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for bx, by in train_loader:
                bx = bx.to(self.device, non_blocking=pin)
                by = by.to(self.device, non_blocking=pin)

                optimizer.zero_grad()
                logits = self.model(bx, return_logits=True).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, by, pos_weight=pos_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_train_loss = epoch_loss / max(1, n_batches)
            current_lr = optimizer.param_groups[0]["lr"]

            # Validation evaluation
            val_pr_auc = 0.0
            val_roc_auc = 0.5
            val_loss = 0.0

            if X_val is not None and y_val is not None:
                val_probs = self._predict_proba_loop(X_val, batch_size=batch_size * 2)[:, 1]
                if len(np.unique(y_val)) > 1:
                    p, r, _ = precision_recall_curve(y_val, val_probs)
                    val_pr_auc = float(auc(r, p))
                    try:
                        val_roc_auc = float(roc_auc_score(y_val, val_probs))
                    except Exception:
                        val_roc_auc = 0.5
                else:
                    val_pr_auc = 0.0

                scheduler.step(val_pr_auc)

                if val_pr_auc > best_pr_auc:
                    best_pr_auc = val_pr_auc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        logger.info(f"[AttentiveTabResNetWrapper] Early stopping triggered at epoch {epoch + 1}")
                        break

            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_pr_auc"].append(val_pr_auc)
            self.training_history["val_roc_auc"].append(val_roc_auc)
            self.training_history["learning_rate"].append(current_lr)

        # Restore best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_val_pr_auc = best_pr_auc

        # Calibrate decision threshold on validation set
        if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            val_probs = self._predict_proba_loop(X_val, batch_size=batch_size * 2)[:, 1]
            self.optimal_threshold = calibrate_decision_threshold(y_val, val_probs)
        else:
            self.optimal_threshold = 0.5

        return self

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        max_epochs: int = 30,
        batch_size: int = 128,
        patience: int = 7,
    ) -> "AttentiveTabResNetWrapper":
        """
        Trains AttentiveTabResNet with validation PR-AUC early stopping.
        Automatically catches CUDA errors/OOM and falls back to multi-core CPU parallel execution.
        """
        X_train_arr = self._sanitize_array(X_train)
        y_train_arr = np.asarray(y_train, dtype=np.float32).ravel()

        X_val_arr = self._sanitize_array(X_val) if X_val is not None else None
        y_val_arr = np.asarray(y_val, dtype=np.float32).ravel() if y_val is not None else None

        if self.model is None or self.num_continuous is None:
            total_cols = X_train_arr.shape[1]
            cat_cols = len(self.cat_cardinalities)
            self.num_continuous = max(0, total_cols - cat_cols)
            self._build_model()

        try:
            return self._fit_loop(X_train_arr, y_train_arr, X_val_arr, y_val_arr, max_epochs, batch_size, patience)
        except Exception as e:
            if self.device.type == "cuda":
                logger.warning(
                    f"[AttentiveTabResNetWrapper] CUDA training encountered: {e}. "
                    "Gracefully falling back to multi-core CPU parallel processing..."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                try:
                    torch.set_num_threads(os.cpu_count() or 4)
                except Exception:
                    pass
                assert self.model is not None
                self.model = self.model.to(self.device)
                return self._fit_loop(X_train_arr, y_train_arr, X_val_arr, y_val_arr, max_epochs, batch_size, patience)
            else:
                raise e

    def _predict_proba_loop(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        all_p1 = []

        with torch.no_grad():
            N = len(X)
            for i in range(0, N, batch_size):
                chunk = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=self.device)
                p1_chunk = self.model(chunk, return_logits=False).squeeze(-1).cpu().numpy().flatten()
                all_p1.append(p1_chunk)

        p1 = np.concatenate(all_p1) if all_p1 else np.empty(0, dtype=np.float32)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict_proba(self, X: Any, batch_size: int = 1024) -> np.ndarray:
        """
        Computes binary probability distribution [P(On-time), P(Late Delivery Risk)].
        """
        X_arr = self._sanitize_array(X)
        try:
            return self._predict_proba_loop(X_arr, batch_size=batch_size)
        except Exception as e:
            if self.device.type == "cuda":
                logger.warning(
                    f"[AttentiveTabResNetWrapper] CUDA inference failed: {e}. "
                    "Falling back to multi-core CPU inference..."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                assert self.model is not None
                self.model = self.model.to(self.device)
                return self._predict_proba_loop(X_arr, batch_size=batch_size)
            else:
                raise e

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        """
        Binary prediction using calibrated decision threshold.
        """
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thresh).astype(int)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves model weights and wrapper metadata to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        payload = {
            "model_state": self.model.state_dict(),
            "num_continuous": self.num_continuous,
            "cat_cardinalities": self.cat_cardinalities,
            "d_token": self.d_token,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_res_blocks": self.num_res_blocks,
            "dropout_rate": self.dropout_rate,
            "optimal_threshold": self.optimal_threshold,
            "best_val_pr_auc": self.best_val_pr_auc,
            "training_history": self.training_history,
        }
        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[Union[str, torch.device]] = None) -> "AttentiveTabResNetWrapper":
        """Loads fitted AttentiveTabResNetWrapper checkpoint from disk."""
        filepath = Path(filepath)
        payload = torch.load(filepath, map_location="cpu")
        wrapper = cls(
            num_continuous=payload["num_continuous"],
            cat_cardinalities=payload["cat_cardinalities"],
            d_token=payload["d_token"],
            num_heads=payload["num_heads"],
            hidden_dim=payload["hidden_dim"],
            num_res_blocks=payload["num_res_blocks"],
            dropout_rate=payload["dropout_rate"],
            device=device,
        )
        assert wrapper.model is not None
        wrapper.model.load_state_dict(payload["model_state"])
        wrapper.optimal_threshold = payload.get("optimal_threshold", 0.5)
        wrapper.best_val_pr_auc = payload.get("best_val_pr_auc", -1.0)
        wrapper.training_history = payload.get("training_history", {})
        return wrapper
