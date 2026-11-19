"""
PyTorch Dataset and DataLoader abstractions for pharmaceutical supply chain forecasting.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.config import BATCH_SIZE


class HSCMDataset(Dataset):
    """PyTorch Dataset wrapper for pharmaceutical tabular records."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Generates standard train, validation, and test PyTorch DataLoaders."""
    train_dataset = HSCMDataset(X_train, y_train)
    val_dataset = HSCMDataset(X_val, y_val)
    test_dataset = HSCMDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader
