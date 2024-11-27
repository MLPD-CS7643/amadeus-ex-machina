import torch.nn as nn
import torch


class RNNModel(nn.Module):
    """
    A simple RNN model for sequential data processing.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN model.

        Args:
            input_size (int): Number of features in the input vector.
            hidden_size (int): Number of features in the hidden state.
            output_size (int): Number of features in the output vector.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize layers
        self.hidden_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state=None):
        # Handle 2D input (batch_size, input_size)
        if input.dim() == 2:
            input = input.unsqueeze(1)  # Add seq_len=1, shape becomes (batch_size, 1, input_size)

        batch_size, seq_len, _ = input.size()

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=input.device)

        outputs = []
        for t in range(seq_len):
            # Select the input for the current time step
            x_t = input[:, t, :]

            # Concatenate input and hidden state
            conc = torch.cat((x_t, hidden_state), dim=1)

            # Update hidden state
            hidden_state = self.tanh(self.hidden_layer(conc))

            # Compute output
            output_t = self.log_softmax(self.output_layer(hidden_state))
            outputs.append(output_t)

        # Stack outputs along the time dimension
        outputs = torch.stack(outputs, dim=1)

        # If the original input was 2D, return a 2D output
        if outputs.size(1) == 1:
            outputs = outputs.squeeze(1)

        return outputs  # Return only outputs