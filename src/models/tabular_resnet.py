"""
Tabular ResNet with Learned Entity Embeddings for High-Cardinality Categoricals.
Includes GPU Acceleration (CUDA pinned memory, non-blocking streaming, seed sync)
and Automatic Multi-Core CPU Fallback on Error or OOM.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, auc

from src.config import DEVICE

logger = logging.getLogger(__name__)


class TabularResBlock(nn.Module):
    """Residual block with batch normalization, ReLU activation, and dropout."""

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabResNet(nn.Module):
    """
    Modern Tabular ResNet with categorical entity embeddings.
    """

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cat_cardinalities = cat_cardinalities

        # Entity embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card + 2, embedding_dim=min(32, max(4, (card + 1) // 2)))
            for card in cat_cardinalities
        ])
        total_embed_dim = sum(e.embedding_dim for e in self.embeddings)

        # Continuous projection
        self.cont_proj = (
            nn.Sequential(
                nn.BatchNorm1d(num_continuous),
                nn.Linear(num_continuous, hidden_dim // 2),
                nn.ReLU(),
            )
            if num_continuous > 0
            else None
        )
        cont_dim = hidden_dim // 2 if num_continuous > 0 else 0

        # Input fusion
        in_dim = cont_dim + total_embed_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        # Residual stack
        self.blocks = nn.ModuleList([TabularResBlock(hidden_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)])

        # Classifier head
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into continuous and categorical
        x_cont = x[:, : self.num_continuous]
        x_cat = x[:, self.num_continuous :].long()

        embed_outs = []
        for i, emb in enumerate(self.embeddings):
            # Clamp to vocabulary bounds
            cat_idx = torch.clamp(x_cat[:, i], 0, emb.num_embeddings - 1)
            embed_outs.append(emb(cat_idx))

        if embed_outs:
            cat_rep = torch.cat(embed_outs, dim=1)
        else:
            cat_rep = torch.empty((x.size(0), 0), device=x.device)

        if self.cont_proj is not None and x_cont.size(1) > 0:
            cont_rep = self.cont_proj(x_cont)
            fused = torch.cat([cont_rep, cat_rep], dim=1)
        else:
            fused = cat_rep

        h = self.in_proj(fused)
        for b in self.blocks:
            h = b(h)

        out = self.head(h)
        return out


class TabResNetWrapper:
    """Wrapper providing GPU-accelerated fit, predict_proba, and graceful CPU fallback for TabResNet."""

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scale_pos_weight: float = 8.0,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
    ):
        # Resolve device
        if device is None:
            self.device = DEVICE
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.seed = seed
        torch.manual_seed(seed)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.active_device = self.device.type
        self.fallback_occurred = False

        self.num_continuous = num_continuous
        self.cat_cardinalities = cat_cardinalities
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.scale_pos_weight = scale_pos_weight

        self.model = TabResNet(
            num_continuous=num_continuous,
            cat_cardinalities=cat_cardinalities,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
        ).to(self.device)

    def _fit_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        max_epochs: int,
        batch_size: int,
        patience: int,
    ) -> "TabResNetWrapper":
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_w = torch.tensor([self.scale_pos_weight], device=self.device)

        pin = (self.device.type == "cuda")
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)

        best_pr_auc = -1.0
        best_state = None
        patience_ctr = 0

        for epoch in range(max_epochs):
            self.model.train()
            for bx, by in train_loader:
                bx = bx.to(self.device, non_blocking=pin)
                by = by.to(self.device, non_blocking=pin)
                optimizer.zero_grad()
                out = self.model(bx).squeeze(-1)
                eps = 1e-7
                probs = torch.clamp(out, eps, 1.0 - eps)
                loss = -(pos_w * by * torch.log(probs) + (1.0 - by) * torch.log(1.0 - probs)).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) > 1:
                    p, r, _ = precision_recall_curve(y_val, val_probs)
                    val_pr_auc = float(auc(r, p))
                else:
                    val_pr_auc = 0.0

                if val_pr_auc > best_pr_auc:
                    best_pr_auc = val_pr_auc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: int = 20,
        batch_size: int = 128,
        patience: int = 5,
    ) -> "TabResNetWrapper":
        try:
            return self._fit_loop(X_train, y_train, X_val, y_val, max_epochs, batch_size, patience)
        except Exception as e:
            if self.device.type == "cuda":
                logger.warning(
                    f"[TabResNetWrapper] CUDA training failed with error: {e}. "
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
                self.model = self.model.to(self.device)
                return self._fit_loop(X_train, y_train, X_val, y_val, max_epochs, batch_size, patience)
            else:
                raise e

    def _predict_proba_loop(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        all_p1 = []
        with torch.no_grad():
            N = len(X)
            for i in range(0, N, batch_size):
                chunk = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=self.device)
                p1_chunk = self.model(chunk).cpu().numpy().flatten()
                all_p1.append(p1_chunk)

        p1 = np.concatenate(all_p1) if all_p1 else np.empty(0, dtype=np.float32)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        try:
            return self._predict_proba_loop(X, batch_size=batch_size)
        except Exception as e:
            if self.device.type == "cuda":
                logger.warning(
                    f"[TabResNetWrapper] CUDA inference failed with error: {e}. "
                    "Falling back to CPU inference..."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                return self._predict_proba_loop(X, batch_size=batch_size)
            else:
                raise e
