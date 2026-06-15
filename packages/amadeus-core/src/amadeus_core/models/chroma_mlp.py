"""
Compact per-frame chroma classifier.

Input is a single 24-dim chroma frame; output is class logits. A lightweight,
self-consistent baseline (trained on the same ``chroma24_from_audio`` features used at
inference, so there is no train/inference feature mismatch).
"""

import torch
import torch.nn as nn


class ChromaMLP(nn.Module):
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
