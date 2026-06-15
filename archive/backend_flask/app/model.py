"""
Checkpoint-faithful CRNN chord classifier.

IMPORTANT: This architecture intentionally mirrors the *saved weights* in
``models/checkpoints/CRNNModel_best_model.pth`` -- NOT the current
``models/CRNN.py`` in the repo root (which was rewritten to a ``cnn_blocks``
ModuleList and will not load this checkpoint).

The checkpoint was produced by an earlier CRNN revision recoverable from git
history (commits ``b12817b`` / ``0f3fca6``):

    Conv2d(1, 64, k3, pad1) -> ReLU -> AdaptiveMaxPool2d((None, 1))
    Conv2d(64, 128, k3, pad1) -> ReLU -> AdaptiveMaxPool2d((None, 1))
    -> flatten (128 * input_features) -> bi-LSTM(hidden=4) -> Linear(8, num_classes)

It was trained on tabular Billboard NNLS chroma (24-dim per frame), so each input
row is treated as a length-1 sequence -- i.e. it is effectively a per-frame
classifier. Verified to load the checkpoint and emit 454-class logits.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults that match the shipped checkpoint.
CHECKPOINT_INPUT_FEATURES = 24
CHECKPOINT_HIDDEN_SIZE = 4

# Models whose forward expects a sequence (batch, seq_length, input_dim) and emit one
# label per window -- inference must feed sliding windows rather than single frames.
SEQUENCE_MODELS = {"ChromaTransformerModel"}


class CRNNChordModel(nn.Module):
    def __init__(
        self,
        input_features: int = CHECKPOINT_INPUT_FEATURES,
        num_classes: int = 454,
        hidden_size: int = CHECKPOINT_HIDDEN_SIZE,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Convolutional feature extractor.
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Reduce the (singleton) time axis to 1 while preserving the frequency axis.
        self.pool = nn.AdaptiveMaxPool2d((None, 1))

        # LSTM input is channels * freq_bins after the conv stack (128 * input_features).
        self.lstm = nn.LSTM(
            input_size=128 * input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, input_features)`` -- one chroma frame per row.
        Returns:
            ``(batch, num_classes)`` logits.
        """
        # (batch, features) -> (batch, 1, features, 1) for the 2-D conv stack.
        x = x.unsqueeze(1).unsqueeze(-1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # (batch, channels, freq, time=1) -> (batch, time=1, channels*freq)
        batch_size, channels, freq_bins, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, -1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last (only) timestep
        return self.fc(x)


class ChromaMLP(nn.Module):
    """
    Compact per-frame chroma classifier (the retrained, self-consistent model).

    Input is a single 24-dim chroma frame; output is class logits. Trained on
    GuitarSet audio whose features come from the SAME ``chroma24_from_audio`` used at
    inference, so there is no train/inference feature mismatch.
    """

    def __init__(
        self,
        input_features: int = 24,
        num_classes: int = 2,
        hidden_sizes=(256, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_features = input_features
        layers = []
        prev = input_features
        for hidden in hidden_sizes:
            layers += [
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    """
    Transformer encoder over a window of chroma frames -> one chord label.

    Ported from the repo's ``models/ChromaTransformer.py`` (the team's best-performing
    architecture) so the backend stays self-contained. Input is
    ``(batch, seq_length, input_dim)``; attention across the window supplies the
    temporal context a per-frame model lacks.
    """

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
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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


# Registry so artifacts can name the architecture to reconstruct (see build_model).
_MODEL_REGISTRY = {
    "ChromaMLP": ChromaMLP,
    "CRNNChordModel": CRNNChordModel,
    "ChromaTransformerModel": ChromaTransformerModel,
}


def build_model(config: dict) -> nn.Module:
    """
    Construct a model from a saved ``model_config.json`` dict.

    Required key: ``model_class`` (and ``num_classes``). Other keys are
    architecture-specific and fall back to sensible defaults.
    """
    name = config.get("model_class")
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_class: {name!r}")

    if name == "ChromaMLP":
        return ChromaMLP(
            input_features=int(config.get("input_features", 24)),
            num_classes=int(config["num_classes"]),
            hidden_sizes=tuple(config.get("hidden_sizes", (256, 256))),
            dropout=float(config.get("dropout", 0.3)),
        )
    if name == "ChromaTransformerModel":
        return ChromaTransformerModel(
            input_dim=int(config.get("input_features", config.get("input_dim", 24))),
            seq_length=int(config["seq_length"]),
            num_classes=int(config["num_classes"]),
            d_model=int(config.get("d_model", 128)),
            nhead=int(config.get("nhead", 8)),
            num_layers=int(config.get("num_layers", 4)),
            dim_feedforward=int(config.get("dim_feedforward", 1024)),
            dropout=float(config.get("dropout", 0.1)),
        )
    # CRNNChordModel
    return CRNNChordModel(
        input_features=int(config.get("input_features", 24)),
        num_classes=int(config["num_classes"]),
        hidden_size=int(config.get("hidden_size", CHECKPOINT_HIDDEN_SIZE)),
    )
