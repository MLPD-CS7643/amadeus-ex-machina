"""
Checkpoint-faithful CRNN chord classifier (legacy).

This architecture intentionally mirrors the *saved weights* in the original
``CRNNModel_best_model.pth`` Billboard checkpoint -- NOT the research ``CRNN.py``.
It was trained on tabular Billboard NNLS chroma (24-dim per frame), so each input row
is treated as a length-1 sequence -- effectively a per-frame classifier.

    Conv2d(1, 64, k3, pad1) -> ReLU -> AdaptiveMaxPool2d((None, 1))
    Conv2d(64, 128, k3, pad1) -> ReLU -> AdaptiveMaxPool2d((None, 1))
    -> flatten (128 * input_features) -> bi-LSTM(hidden=4) -> Linear(8, num_classes)

Retained only as a fallback for that legacy checkpoint; new models use
``ChromaTransformerModel`` / ``ChromaMLP``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults that match the shipped legacy checkpoint.
CHECKPOINT_INPUT_FEATURES = 24
CHECKPOINT_HIDDEN_SIZE = 4


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
