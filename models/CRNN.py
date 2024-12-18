import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNModel(nn.Module):
    def __init__(
        self, input_features, num_classes, hidden_size, num_layers=1, bidirectional=True, **cnn_params
    ):
        super(CRNNModel, self).__init__()

        self.input_features = input_features
        self.n_blocks = cnn_params.pop("n_blocks", 3)
        self.block_depth = cnn_params.pop("block_depth", 3)
        self.pad = cnn_params.pop("pad", 1)
        self.stride = cnn_params.pop("stride", 1)
        self.k_conv = cnn_params.pop("k_conv", 3)
        self.dropout = cnn_params.pop("dropout", 0.2)
        self.out_channels = cnn_params.pop("out_channels", 64)

        # Convolutional layers
        self.cnn_blocks = nn.ModuleList()
        for n in range(self.n_blocks):
            scale = 2**n
            if n == 0:
                in_ch = 1
            else:
                in_ch = self.out_channels * 2**(n-1)
            block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=self.out_channels*scale, kernel_size=self.k_conv, padding=self.pad, stride=self.stride, bias=False),
            nn.BatchNorm2d(self.out_channels*scale),
            nn.ReLU(inplace=True),
            )
            for _ in range(self.block_depth-1):
                block.append(nn.Conv2d(in_channels=self.out_channels*scale, out_channels=self.out_channels*scale, kernel_size=self.k_conv, padding=self.pad, stride=self.stride, bias=False))
                block.append(nn.BatchNorm2d(self.out_channels*scale))
                block.append(nn.ReLU(inplace=True))
            self.cnn_blocks.append(block)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        #self.lstm_input_size = int((self.out_channels * 2**(self.n_blocks-1))/4)

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
        x = torch.zeros((1, 1, self.input_features, 1))  # Single feature
        with torch.no_grad():
            for block in self.cnn_blocks:
                x = block(x)
            if self.dropout > 0:
                x = self.dropout_layer(x)
            x = self.avg_pool(x)
        return x.size(1) * x.size(2)

    def forward(self, x):
        # x: (batch_size, seq_length, input_features)

        batch_size, seq_length, _ = x.size()
        # Flatten seq dimension into batch to apply CNN per-chord
        x = x.view(batch_size * seq_length, self.input_features)  # (B*seq, input_features)

        # CNN expects: (B*seq, 1, freq, 1)
        x = x.unsqueeze(1).unsqueeze(-1)
        for block in self.cnn_blocks:
            x = block(x)
        if self.dropout > 0:
            x = self.dropout_layer(x)
        x = self.avg_pool(x)

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