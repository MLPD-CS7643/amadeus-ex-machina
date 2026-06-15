"""
Transformer encoder over a window of chroma frames -> one chord label.

This is the team's best-performing architecture and the one shipped in the default
artifacts. Input is ``(batch, seq_length, input_dim)``; attention across the window
supplies the temporal context a per-frame model lacks. Layer names are kept stable so
existing ``model.pt`` checkpoints load unchanged.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)


class ChromaTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 24,
        seq_length: int = 16,
        num_classes: int = 25,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout, max_len=seq_length
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_length, input_dim)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # pool over the window
        return self.classifier(x)
