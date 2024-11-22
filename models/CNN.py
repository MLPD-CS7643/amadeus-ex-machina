import torch.nn as nn
import torch
import numpy as np


class CNNModel(nn.Module):
    # An CNN model

    def __init__(self):
        """Initialize the CNN model
        Args:
            None

        Returns:
            None
        """
        super().__init__()

        # conv layer
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0
        )

        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv output: (32 - 7) / 1 + 1 = 26 x 26
        # pooling output: 26 / 2 = 13 x 13
        # filters: 64
        # classes: 10
        self.fc = nn.Linear(64 * 13 * 13, 10)

    def forward(self, x):

        # conv layer
        x = self.conv1(x)

        # relu
        x = torch.relu(x)

        # max pooling layer
        x = self.pool(x)

        # flatten input through fully connected layer
        return self.fc(torch.flatten(x, 1))
