import torch.nn as nn
import torch


class RNNModel(nn.Module):
    # An RNN model

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the RNN model
        Args:
            input_size (int): the number of features in the inputs.
            hidden_size (int): the size of the hidden layer
            output_size (int): the size of the output layer

        Returns:
            None
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize block that creates hidden
        self.hidden_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        # initialize block that creates output
        self.output_layer = nn.Linear(input_size + hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Initialize the hidden state as None
        self.hidden_state = None

    def forward(self, input):
        """Forward pass of RNN
        Args:
            input (tensor): a batch of data of shape (batch_size, input_size) at one time step

        Returns:
            output (tensor): the output tensor of shape (batch_size, output_size)
        """
        # Automatically initialize the hidden state if it's not set
        if self.hidden_state is None or self.hidden_state.size(0) != input.size(0):
            self.hidden_state = torch.zeros(input.size(0), self.hidden_size, device=input.device)

        # Flatten input if it has more than 2 dimensions
        if input.dim() > 2:
            input = input.view(input.size(0), -1)

        # Validate input size
        assert input.size(1) == self.input_size, \
            f"Expected input size {self.input_size}, but got {input.size(1)}."

        # Concatenate input vector and hidden state
        conc = torch.cat((input, self.hidden_state), dim=1)

        # Update hidden state
        self.hidden_state = self.tanh(self.hidden_layer(conc))

        # Compute output
        output = self.log_softmax(self.output_layer(conc))

        return output
