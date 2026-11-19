"""
1D Convolutional Neural Network (1D-CNN) for High-Throughput Tabular Feature Extraction.
Optimized for ultra-low latency edge/cloud inference in pharmaceutical logistics.
"""

import math
import torch
import torch.nn as nn


class ExactAdaptiveAvgPool1d(nn.Module):
    """
    Exact mathematical equivalent of nn.AdaptiveAvgPool1d that exports cleanly to ONNX
    regardless of whether input length is a factor of output length.
    """

    def __init__(self, output_size: int = 16):
        super().__init__()
        self.output_size = output_size
        self._W_cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_len = x.size(-1)
        if in_len == self.output_size:
            return x
        if in_len % self.output_size == 0:
            return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size)

        key = (in_len, self.output_size, str(x.device))
        if key not in self._W_cache:
            W = torch.zeros(in_len, self.output_size, dtype=x.dtype, device=x.device)
            for i in range(self.output_size):
                s = math.floor(i * in_len / self.output_size)
                e = math.ceil((i + 1) * in_len / self.output_size)
                W[s:e, i] = 1.0 / (e - s)
            self._W_cache[key] = W
        return torch.matmul(x, self._W_cache[key])


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
            ExactAdaptiveAvgPool1d(output_size=16),
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
