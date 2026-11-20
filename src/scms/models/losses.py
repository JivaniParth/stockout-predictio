"""
Loss function implementations for imbalanced clinical stockout prediction:
1. Weighted Binary Cross-Entropy (WeightedBCE)
2. Standard Unweighted Binary Cross-Entropy (BCE)
3. Binary Focal Loss (Lin et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss to penalize False Negatives (missed stockouts).
    """

    def __init__(self, pos_weight: float = 6.0, eps: float = 1e-7):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        loss = -(
            self.pos_weight * y_true * torch.log(y_pred)
            + (1.0 - y_true) * torch.log(1.0 - y_pred)
        )
        return torch.mean(loss)


class UnweightedBCELoss(nn.Module):
    """
    Numerically stable AMP-safe Unweighted Binary Cross-Entropy Loss with epsilon clamping.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.clamp(y_pred.float(), self.eps, 1.0 - self.eps)
        y_true = y_true.float()
        loss = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
        return torch.mean(loss)


class FocalLoss(nn.Module):
    """
    Binary Focal Loss: down-weights easy negatives to focus gradients on hard late-delivery consignments.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = -focal_weight * torch.log(p_t)
        return torch.mean(loss)


def get_loss_function(variant: str = "weighted_bce", pos_weight: float = 6.0, alpha: float = 0.75, gamma: float = 2.0) -> nn.Module:
    """Factory creating configured loss function."""
    variant = variant.lower()
    if variant in ["weighted_bce", "weighted"]:
        return WeightedBCELoss(pos_weight=pos_weight)
    elif variant in ["unweighted_bce", "bce"]:
        return UnweightedBCELoss()
    elif variant in ["focal", "focal_loss"]:
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif variant in ["binary_focal", "binary_focal_loss"]:
        from src.models.dl_trainer import BinaryFocalLoss
        return BinaryFocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss variant: '{variant}'. Must be weighted_bce, unweighted_bce, focal_loss, or binary_focal.")


# Alias BinaryFocalLoss from dl_trainer for unified imports
def __getattr__(name: str):
    if name == "BinaryFocalLoss":
        from src.models.dl_trainer import BinaryFocalLoss
        return BinaryFocalLoss
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

