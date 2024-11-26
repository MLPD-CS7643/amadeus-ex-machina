import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNModel(nn.Module):
    def __init__(
        self, input_channels, num_classes, hidden_size, num_layers=1, bidirectional=True
    ):
        """
        Initialize the CRNN model for audio classification.

        Args:
            input_channels (int): Number of channels in the input (e.g., 2 for stereo spectrograms).
            num_classes (int): Number of output classes.
            hidden_size (int): The size of the LSTM hidden state.
            num_layers (int): Number of LSTM layers (default: 1).
            bidirectional (bool): If True, use a bidirectional LSTM (default: True).
        """
        super(CRNNModel, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate LSTM input size based on convolutional output
        # self.lstm_input_size = self._calculate_lstm_input_size(input_channels)
        # print(f"Calculated LSTM input size: {self.lstm_input_size}")

        # LSTM layers for temporal feature extraction
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=2048,  # Dynamically calculated size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Fully connected layer for classification
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def _calculate_lstm_input_size(self, input_channels):
        """
        Calculate the LSTM input size based on the output of the convolutional layers.
        """
        # Dummy input with batch_size=1, time_steps=1, and frequency_bins=128
        dummy_input = torch.zeros((1, input_channels, 128, 128))
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
        # Flatten frequency bins and channels
        lstm_input_size = x.size(1) * x.size(2)
        return lstm_input_size


    def forward(self, x):
        """
        Forward pass of the CRNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, frequency_bins, time_steps).

        Returns:
            torch.Tensor: Output tensor with predicted class scores.
        """
        # Convolutional layers with ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Reshape to prepare for LSTM (batch_size, time_steps, features)
        batch_size, channels, freq_bins, time_steps = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch_size, time_steps, channels, frequency_bins)
        x = x.contiguous().view(batch_size, time_steps, -1)  # Flatten

        # LSTM layer
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size,
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size,
        ).to(x.device)
        x, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        x = x[:, -1, :]

        # Fully connected layer
        x = self.fc(x)

        return x
