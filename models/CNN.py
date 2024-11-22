import torch.nn as nn
import torch
import numpy as np


class CNNModel(nn.Module):
    # A CNN model
    def __init__(self, input_channels, num_classes):
        """Initialize the CNN model"""
        super(CNNModel, self).__init__()

        # Convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=64, kernel_size=7, stride=1, padding=0
        )

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Placeholder for the fully connected layer
        self.fc = None
        self.num_classes = num_classes

    def forward(self, x):
        # Convolutional layer
        x = self.conv1(x)
        x = torch.relu(x)

        # Max pooling layer
        x = self.pool(x)

        # Flatten input through fully connected layer
        if self.fc is None:  # Initialize self.fc dynamically
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fc = nn.Linear(flattened_size, self.num_classes).to(x.device)

        x = self.fc(torch.flatten(x, 1))
        return x
