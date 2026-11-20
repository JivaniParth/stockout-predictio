"""
PyTorch Model Training Pipeline with Class-Weighted Loss, Early Stopping, and Threshold Calibration.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

from src.config import (
    DEVICE,
    MAX_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    MODELS_DIR,
)


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        pos_weight: float = 8.0,
        device: torch.device = DEVICE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.pos_weight = torch.tensor([pos_weight], device=device)
        self.criterion = nn.BCELoss(weight=None) # We apply class weighting dynamically or via weighted BCE
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_val_score = -float("inf")
        self.best_checkpoint_path = MODELS_DIR / f"{model_name}_best.pt"
        self.optimal_threshold = 0.5

    def _weighted_bce(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Weighted BCE loss penalizing false negatives for stockout risk."""
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        loss = - (self.pos_weight * y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
        return torch.mean(loss)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self._weighted_bce(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self._weighted_bce(y_pred, y_batch)
                total_loss += loss.item() * len(X_batch)

                all_preds.extend(y_pred.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        val_loss = total_loss / len(dataloader.dataset)
        preds = np.array(all_preds)
        targets = np.array(all_targets)

        # Calculate ROC-AUC and PR-AUC
        roc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
        precisions, recalls, _ = precision_recall_curve(targets, preds)
        pr_auc = auc(recalls, precisions)

        return val_loss, roc, pr_auc, preds, targets

    def find_optimal_threshold(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """Finds decision threshold on validation set that maximizes F1-score."""
        best_thresh = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.1, 0.9, 81):
            bin_preds = (preds >= t).astype(int)
            f1 = f1_score(targets, bin_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        self.optimal_threshold = float(best_thresh)
        return self.optimal_threshold

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = MAX_EPOCHS,
        patience: int = EARLY_STOPPING_PATIENCE,
    ) -> Dict[str, Any]:
        """Executes full training with early stopping on validation PR-AUC."""
        history = {"train_loss": [], "val_loss": [], "val_roc": [], "val_pr_auc": []}
        patience_counter = 0

        print(f"[*] Training {self.model_name} on {self.device}...")
        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss, v_roc, v_pr_auc, val_preds, val_targets = self.evaluate(val_loader)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["val_roc"].append(v_roc)
            history["val_pr_auc"].append(v_pr_auc)

            # Composite metric for early stopping (PR-AUC is vital for imbalanced stockouts)
            score = v_pr_auc

            if score > self.best_val_score:
                self.best_val_score = score
                self.find_optimal_threshold(val_preds, val_targets)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimal_threshold": self.optimal_threshold,
                        "best_val_score": self.best_val_score,
                    },
                    self.best_checkpoint_path,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or patience_counter == 0:
                print(
                    f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
                    f"Val ROC: {v_roc:.4f} | Val PR-AUC: {v_pr_auc:.4f} | Best Thresh: {self.optimal_threshold:.2f}"
                )

            if patience_counter >= patience:
                print(f"  [!] Early stopping triggered at epoch {epoch}.")
                break

        # Load best weights
        checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimal_threshold = checkpoint["optimal_threshold"]
        print(f"  [+] Loaded best checkpoint with Val PR-AUC: {self.best_val_score:.4f}, Optimal Threshold: {self.optimal_threshold:.2f}")
        return history
