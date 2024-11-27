import torch.nn as nn
import torch


class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        """
        Initialize the CNN model.
        Args:
            input_channels (int): Number of input features per sample.
            num_classes (int): Number of output classes.
        """
        super(CNNModel, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=1,  # Single channel after reshaping
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layer (to be initialized dynamically)
        self.fc = None

    def forward(self, x):
        """
        Forward pass of the CNN model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels).
        Returns:
            torch.Tensor: Output tensor with predicted class scores.
        """
        # Add a channel dimension for Conv1d
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, input_channels)

        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the tensor and initialize the fully connected layer dynamically
        if self.fc is None:
            # Calculate the flattened size
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fc = nn.Linear(flattened_size, self.num_classes).to(x.device)

        # Flatten and pass through the fully connected layer
        x = self.fc(torch.flatten(x, 1))

        return x