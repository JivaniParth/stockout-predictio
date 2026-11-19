"""
Deep Residual Multi-Layer Perceptron (Res-MLP) for Tabular HSCM Prediction.
Features skip-connections, LayerNorm, and GELU activations for robust tabular representation learning.
"""

import torch
import torch.nn as nn


class TabularResBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x + self.block(x))


class DeepTabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_block1 = TabularResBlock(hidden_dim, dropout_rate)
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_block2 = TabularResBlock(hidden_dim // 2, dropout_rate)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res_block1(h)
        h = self.transition(h)
        h = self.res_block2(h)
        logits = self.head(h)
        return torch.sigmoid(logits)
