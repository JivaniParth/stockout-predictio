"""
Shared Deep Learning Trainer Module.
Features:
- AdamW optimizer with weight decay
- Learning rate scheduler (ReduceLROnPlateau)
- Gradient clipping (torch.nn.utils.clip_grad_norm_)
- Automatic Mixed Precision (AMP) on CUDA with seamless CPU fallback
- Early stopping monitored on validation PR-AUC
- Configurable loss variants (WeightedBCE, BCE, FocalLoss)
- Fixed seed reproducibility across 5 seeds with per-seed metric checkpointing

Validation Partition Policy:
By default, the 2014 prospective validation partition is used for BOTH:
1. Early stopping (epoch checkpoint selection)
2. Decision threshold calibration (tau* selection maximizing F1 on delays)
An optional decoupled mode divides 2014 into val-A (early stopping) and val-B (threshold calibration)
to evaluate and prevent validation overfitting.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

from src.models.losses import get_loss_function


class SharedTrainer:
    """
    Unified training engine for all PyTorch neural architectures in the project.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        loss_variant: str = "weighted_bce",
        pos_weight: float = 6.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        use_amp: bool = True,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        seed: int = 42,
        require_gpu: bool = False,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if require_gpu and self.device.type != "cuda":
            raise RuntimeError(f"--require-gpu failed loudly: GPU requested for {model_name} but resolved to {self.device}.")
        self.model = model.to(self.device)
        self.model_name = model_name
        self.gradient_clip = gradient_clip
        self.seed = seed
        self.use_amp = use_amp and (self.device.type == "cuda")

        self.criterion = get_loss_function(
            variant=loss_variant,
            pos_weight=pos_weight,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.checkpoint_dir = Path(checkpoint_dir or "models")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / f"{model_name}_seed_{seed}_best.pt"

        self.best_val_score = -float("inf")
        self.optimal_threshold = 0.5
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_pr_auc": [],
            "val_roc_auc": [],
            "learning_rate": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        pin = (self.device.type == "cuda")

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device, non_blocking=pin)
            y_batch = y_batch.to(self.device, non_blocking=pin)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    y_pred = self.model(X_batch)
                    loss = self.criterion(y_pred.float(), y_batch.float())
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred.float(), y_batch.float())
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.optimizer.step()

            total_loss += loss.item() * len(X_batch)

        return total_loss / max(len(dataloader.dataset), 1)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds: List[float] = []
        all_targets: List[float] = []
        pin = (self.device.type == "cuda")

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device, non_blocking=pin)
                y_batch = y_batch.to(self.device, non_blocking=pin)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        y_pred = self.model(X_batch)
                        loss = self.criterion(y_pred.float(), y_batch.float())
                else:
                    y_pred = self.model(X_batch)
                    loss = self.criterion(y_pred.float(), y_batch.float())

                total_loss += loss.item() * len(X_batch)
                all_preds.extend(y_pred.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        val_loss = total_loss / max(len(dataloader.dataset), 1)
        preds = np.array(all_preds, dtype=np.float32)
        targets = np.array(all_targets, dtype=np.float32)

        roc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
        precisions, recalls, _ = precision_recall_curve(targets, preds)
        pr_auc = auc(recalls, precisions) if len(recalls) > 1 else 0.0

        return val_loss, roc, pr_auc, preds, targets

    def calibrate_threshold(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """Finds decision threshold maximizing validation F1 score."""
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
        val_calibration_loader: Optional[DataLoader] = None,
        max_epochs: int = 40,
        patience: int = 8,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes complete training loop with early stopping on validation PR-AUC.

        If val_calibration_loader is supplied (decoupled val-A / val-B mode):
        - val_loader (val-A) is used solely for early stopping and LR scheduler.
        - val_calibration_loader (val-B) is used solely for threshold calibration.
        Otherwise, val_loader is used for both.
        """
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss, v_roc, v_pr_auc, val_preds, val_targets = self.evaluate(val_loader)

            curr_lr = self.optimizer.param_groups[0]["lr"]
            self.training_history["train_loss"].append(t_loss)
            self.training_history["val_loss"].append(v_loss)
            self.training_history["val_pr_auc"].append(v_pr_auc)
            self.training_history["val_roc_auc"].append(v_roc)
            self.training_history["learning_rate"].append(curr_lr)

            # Step scheduler based on validation PR-AUC
            self.scheduler.step(v_pr_auc)

            # Early stopping check on validation PR-AUC
            if v_pr_auc > self.best_val_score:
                self.best_val_score = v_pr_auc

                # Calibrate threshold
                if val_calibration_loader is not None:
                    _, _, _, cal_preds, cal_targets = self.evaluate(val_calibration_loader)
                    self.calibrate_threshold(cal_preds, cal_targets)
                else:
                    self.calibrate_threshold(val_preds, val_targets)

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimal_threshold": self.optimal_threshold,
                        "best_val_pr_auc": self.best_val_score,
                        "seed": self.seed,
                        "model_name": self.model_name,
                    },
                    self.checkpoint_path,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 5 == 0 or patience_counter == 0):
                print(
                    f"[{self.model_name}|Seed {self.seed}] Epoch {epoch:02d}/{max_epochs:02d} | "
                    f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
                    f"Val PR-AUC: {v_pr_auc:.4f} | tau*: {self.optimal_threshold:.2f}"
                )

            if patience_counter >= patience:
                if verbose:
                    print(f"[{self.model_name}|Seed {self.seed}] Early stopping triggered at epoch {epoch}.")
                break

        # Load best checkpoint weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimal_threshold = float(checkpoint.get("optimal_threshold", 0.5))

        return {
            "best_val_pr_auc": self.best_val_score,
            "optimal_threshold": self.optimal_threshold,
            "checkpoint_path": str(self.checkpoint_path),
            "history": self.training_history,
        }
