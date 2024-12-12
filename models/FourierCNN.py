import torch.nn as nn
import torch
from models.StridedFourier import StridedFourier

class FourierCNN(nn.Module):
    def __init__(self, input_size, kernel_size, stride, fourier_size, num_channels=192):
        """
        Initialize the model based on using a fourier transform as the kernel function
        Args:
            input_size (int): Number of bits from the input
            kernel_size (int): Number of bits kernel should be
            stride (int): Number of bits the kernel will stride for each window
            fourier_size (int): Number of bits the fourier transform will look at, at most kernel size
            num_channels (int): Number of output classes
        """
        super(FourierCNN, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.fourier_size = fourier_size
        self.num_channels = num_channels
        self.num_windows = self.input_size//self.kernel_size

        self.fourier = StridedFourier(self.kernel_size, self.stride, self.fourier_size)
        self.linear = nn.Linear(self.num_windows * self.fourier_size, self.num_channels)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, input_size)
        Returns:
            torch.Tensor: Output tensor with predicted class scores
        """

        fouriers = self.fourier(x)
        output = self.linear(fouriers)

        return output

