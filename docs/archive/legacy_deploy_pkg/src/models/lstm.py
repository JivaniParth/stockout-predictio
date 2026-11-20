"""
Bidirectional LSTM with Self-Attention Pooling for Pharmaceutical Supply Chain Prediction.
Models sequential and temporal dependencies with attention-weighted feature aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, hidden_dim)
        weights = F.softmax(self.attn(x), dim=1)  # (B, seq_len, 1)
        pooled = torch.sum(x * weights, dim=1)    # (B, hidden_dim)
        return pooled


class BiLSTMStockoutPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Project flat tabular features into structured sequence of feature tokens
        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, seq_len * embed_dim),
            nn.LayerNorm(seq_len * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Bidirectional Recurrent Encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Attention pooling across sequence steps
        self.attn_pool = AttentionPooling(hidden_dim * 2)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = self.token_proj(x).view(b, self.seq_len, self.embed_dim)
        lstm_out, _ = self.lstm(tokens)
        pooled = self.attn_pool(lstm_out)
        logits = self.classifier(pooled)
        return torch.sigmoid(logits)
