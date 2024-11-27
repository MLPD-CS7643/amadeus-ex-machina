import torch.nn as nn


class MLPChordClassifier(nn.Module):
    def __init__(self, input_size, num_classes, device="cuda"):
        super(MLPChordClassifier, self).__init__()
        self.network = nn.Sequential(
            # Initial feature extraction
            nn.Linear(input_size, 1024, device=device),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Deep feature processing
            nn.Linear(1024, 768, device=device),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(768, 512, device=device),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 384, device=device),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(384, 256, device=device),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 192, device=device),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(192, 128, device=device),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),

            # Final classification layers
            nn.Linear(128, 96, device=device),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(96, num_classes, device=device)
        )

    def forward(self, x):
        x = self.network(x)
        return x
