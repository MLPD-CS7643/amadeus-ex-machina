import torch.nn as nn
import torch

class StridedFourier(nn.Module):
    #module that takes in a sequence from a wav file and applies a fourier transform to windowed views
    def __init__(self, kernel_size, stride, fourier_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.fourier_size = fourier_size

    def forward(self, x):
        # x is of shape (N, L), where N is batch size, L is sequence length
        unfolded = x.unfold(dimension=1, size=self.kernel_size, step=self.stride)  # Shape: (N, number_of_windows, kernel_size)
        
        # Apply a custom function across the last dimension (kernel_size)
        fouriers = torch.fft.fft(unfolded, n=self.fourier_size, dim=-1)  # Shape: (N, number_of_windows, fourier_size)
        
        return fouriers
    
