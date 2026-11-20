"""
Advanced Hybrid LSTM-Transformer Architecture for Pharmaceutical Stockout Prediction.
Synthesizes local recurrent representations (LSTM) with global multi-head self-attention (Transformer).
Directly inspired by Cao et al. (2024) and Lim et al. (2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, ffn_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Self-Attention with residual connection
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        # Position-wise Feed Forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class HybridLSTMTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 6,
        embed_dim: int = 64,
        lstm_hidden_dim: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Tokenization of tabular features
        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, seq_len * embed_dim),
            nn.LayerNorm(seq_len * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Stage 1: Recurrent Encoder for local sequential/temporal correlations
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(lstm_hidden_dim * 2, embed_dim)
        self.recurrent_norm = nn.LayerNorm(embed_dim)

        # Stage 2: Transformer Encoder for global multi-token context
        self.transformer_block = TransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate,
        )

        # Stage 3: Self-Attentive Pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        # 1. Project into tokens: (B, seq_len, embed_dim)
        tokens = self.token_proj(x).view(b, self.seq_len, self.embed_dim)

        # 2. Local Recurrent Feature Processing
        lstm_out, _ = self.lstm(tokens)
        h_recurrent = self.recurrent_norm(tokens + self.lstm_proj(lstm_out))

        # 3. Global Multi-Head Attention Processing
        h_trans = self.transformer_block(h_recurrent)

        # 4. Attentive Pooling
        weights = F.softmax(self.pool_attn(h_trans), dim=1)  # (B, seq_len, 1)
        context = torch.sum(h_trans * weights, dim=1)        # (B, embed_dim)

        # 5. Output Prediction
        logits = self.classifier(context)
        return torch.sigmoid(logits)


class EnhancedHybridTransformer(nn.Module):
    """
    Enhanced Hybrid LSTM-Transformer-MLP with Multi-Head Self-Attention
    and Highway Residual Skip Connections (Cao et al. 2024; Kabir et al. 2025).
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 8,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.tokenizer = nn.Sequential(
            nn.Linear(input_dim, seq_len * embed_dim),
            nn.LayerNorm(seq_len * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.pool_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.skip_proj = nn.Linear(input_dim, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = self.tokenizer(x).view(b, self.seq_len, self.embed_dim)
        lstm_out, _ = self.bilstm(tokens)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        x_att = self.norm1(lstm_out + attn_out)
        ffn_out = self.ffn(x_att)
        x_trans = self.norm2(x_att + ffn_out)
        weights = F.softmax(self.pool_weights(x_trans), dim=1)
        pooled = torch.sum(x_trans * weights, dim=1)
        skip = F.gelu(self.skip_proj(x))
        combined = pooled + skip
        logits = self.head(combined)
        return torch.sigmoid(logits)

