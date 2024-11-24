import torch.nn as nn


class MLPChordClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPChordClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.network(x)
        return x
