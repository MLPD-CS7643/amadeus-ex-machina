import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNModel(nn.Module):
    def __init__(
        self, input_features, num_classes, hidden_size, num_layers=1, bidirectional=True
    ):
        super(CRNNModel, self).__init__()

        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((1, None))  # Pool along frequency axis

        # Placeholder for LSTM and FC layers (initialized dynamically)
        self.lstm = None
        self.fc = None

    def _initialize_lstm_and_fc(self, x):
        """
        Dynamically initialize the LSTM and fully connected layer
        based on the input features from the forward pass.
        """
        batch_size, channels, freq_bins, time_steps = x.size()
        lstm_input_size = freq_bins * channels

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.fc = nn.Linear(lstm_output_size, self.num_classes)

    def forward(self, x):
        # Ensure input has 4 dimensions: (batch_size, channels, freq_bins, time_steps)
        if len(x.shape) == 2:  # (batch_size, input_features)
            x = x.unsqueeze(1).unsqueeze(-1)  # Add channel and time dimensions
        elif len(x.shape) == 3:  # (batch_size, input_features, time_steps)
            x = x.unsqueeze(1)  # Add channel dimension
        elif len(x.shape) != 4:
            raise RuntimeError(
                f"Expected input to be 2D, 3D, or 4D, but got input of shape {x.shape}."
            )

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Dynamically initialize LSTM and FC layers if not already initialized
        if self.lstm is None or self.fc is None:
            self._initialize_lstm_and_fc(x)

        # Reshape for LSTM: (batch_size, time_steps, lstm_input_size)
        batch_size, channels, freq_bins, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_steps, -1)

        # Initialize hidden and cell states on the same device as the input
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size,
            device=x.device,
        )
        c0 = torch.zeros_like(h0)

        # LSTM forward pass
        x, _ = self.lstm(x, (h0, c0))

        # Fully connected layer
        x = self.fc(x[:, -1, :])  # Use the last time step
        return x

    def update_output_layer(self, num_classes):
        """
        Updates the fully connected layer for transfer learning with a new number of classes.

        Args:
            num_classes (int): The number of classes for the new dataset.
        """
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def freeze_feature_extractor(self):
        """
        Freezes the convolutional and LSTM layers for transfer learning.
        """
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        """
        Unfreezes the convolutional and LSTM layers for fine-tuning.
        """
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.conv2.parameters():
            param.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = True


class CRNNModelOld(nn.Module):
    def __init__(
        self, input_features, num_classes, hidden_size, num_layers=1, bidirectional=True
    ):
        super(CRNNModelOld, self).__init__()

        self.input_features = input_features

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((None, 1))  # Reduce time_steps to 1

        # Dynamically calculate LSTM input size
        self.lstm_input_size = self._calculate_lstm_input_size()

        # LSTM layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Fully connected layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def _calculate_lstm_input_size(self):
        dummy_input = torch.zeros((1, 1, self.input_features, 1))  # Single feature
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = self.pool(x)  # Adaptive pooling prevents invalid dimensions
            x = F.relu(self.conv2(x))
            x = self.pool(x)
        return x.size(1) * x.size(2)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)  # Reshape to (batch_size, 1, input_features, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Reshape for LSTM
        batch_size, channels, freq_bins, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, -1)

        # LSTM
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size
        ).to(x.device)
        x, _ = self.lstm(x, (h0, c0))

        # Fully connected layer
        x = self.fc(x[:, -1, :])
        return x