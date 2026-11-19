"""
1D Convolutional Neural Network (1D-CNN) for High-Throughput Tabular Feature Extraction.
Optimized for ultra-low latency edge/cloud inference in pharmaceutical logistics.
"""

import torch
import torch.nn as nn


class Tabular1DCNN(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=16),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, input_dim) -> (B, 1, input_dim)
        x = x.unsqueeze(1)
        h = self.conv1(x)
        h = self.conv2(h)
        h = h.view(h.size(0), -1)
        logits = self.classifier(h)
        return torch.sigmoid(logits)
