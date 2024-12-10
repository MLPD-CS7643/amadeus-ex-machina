import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNModel(nn.Module):
    def __init__(
        self, input_features, num_classes, hidden_size, num_layers=1, bidirectional=True
    ):
        super(CRNNModel, self).__init__()

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
        # x: (batch_size, seq_length, input_features)

        batch_size, seq_length, _ = x.size()
        # Flatten seq dimension into batch to apply CNN per-chord
        x = x.view(batch_size * seq_length, self.input_features)  # (B*seq, input_features)

        # CNN expects: (B*seq, 1, freq, 1)
        x = x.unsqueeze(1).unsqueeze(-1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # After CNN pooling: (B*seq, channels, freq_bins, 1)
        # Flatten to (B*seq, channels*freq_bins)
        b, c, f, t = x.size()  # t should be 1 after pooling
        x = x.view(b, c*f)

        # Reshape back to sequence: (batch_size, seq_length, lstm_input_size)
        x = x.view(batch_size, seq_length, -1)

        # LSTM
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                        batch_size, 
                        self.hidden_size, 
                        device=x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                        batch_size, 
                        self.hidden_size, 
                        device=x.device)

        x, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_length, lstm_output_size)

        # Select the last timestep's output
        x = x[:, -1, :]  # (batch_size, lstm_output_size)

        # Fully connected layer for classification
        x = self.fc(x)  # (batch_size, num_classes)

        return x