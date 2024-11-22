import torch.nn as nn
import torch
import numpy as np


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
        self.hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        # initialize block that creates output
        self.output = nn.Linear(input_size + hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """Forward pass of RNN
        Args:
            input (tensor): a batch of data of shape (batch_size, input_size) at one time step
            hidden (tensor): the hidden value of previous time step of shape (batch_size, hidden_size)

        Returns:
            output (tensor): the output tensor of shape (batch_size, output_size)
            hidden (tensor): the hidden value of current time step of shape (batch_size, hidden_size)
        """

        # concatenate input vector and hidden state
        conc = torch.cat((input, hidden), 1)

        # pass conc through hidden layer + apply tanh
        hidden = self.tanh(self.hidden(conc))

        # pass conc through output layer + apply log softmax
        output = self.log_softmax(self.output(conc))

        return output, hidden
