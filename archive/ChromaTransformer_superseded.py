import torch
import torch.nn as nn
import math


class ChromaTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_length,
        num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super(ChromaTransformerModel, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.d_model = d_model

        # Input embedding layer
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model, dropout, max_len=seq_length
        )

        # Transformer encoder
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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

    def forward(self, x):
        _, seq_length, input_dim = x.size()
        if seq_length != self.seq_length:
            raise ValueError(
                f"Expected seq_length={self.seq_length}, but got {seq_length}"
            )
        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got {input_dim}"
            )

        # Embed input features
        x = self.embedding(x)

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Transpose for transformer
        x = x.transpose(0, 1)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Take the mean of the encoder outputs across the sequence
        x = x.mean(dim=0)

        # Pass through the classification head
        logits = self.classifier(x)

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values dependent on
        # pos and i
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)
