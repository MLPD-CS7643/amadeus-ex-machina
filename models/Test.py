from scipy.io import wavfile
import torch
from torch import nn
import torch.nn.functional as F

samplerate, data = wavfile.read("C:/Users/mattb/Documents/CS7643/Final Project/amadeus-ex-machina/data/emoSynth-DB-fix/emoSynth-DB/all_data/wavs/s1_a0_d1.wav")



class StridedFourier(nn.Module):
    def __init__(self, kernel_size, stride, fourier_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.fourier_size = fourier_size

    def forward(self, x):
        # x is of shape (N, L), where N is batch size, L is sequence length
        unfolded = x.unfold(dimension=1, size=self.kernel_size, step=self.stride)  # Shape: (N, number_of_windows, kernel_size)
        
        # Apply a custom function across the last dimension (kernel_size)
        fouriers = torch.fft.fft(unfolded, n=self.fourier_size, dim=-1)  # Shape: (N, number_of_windows)
        
        return fouriers


class ChordProgressionTranscriber(nn.Module):

    def __init__(self, sample_rate, kernel_size, stride, fourier_size, enc_hidden,dec_hidden, output_size, dropout=0.2):
        super().__init__()
        self.token_size = sample_rate/4
        self.fourier_size = fourier_size
        self.output_size = output_size
        self.strideFourier = StridedFourier(kernel_size, stride, fourier_size)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.LSTM(self.fourier_size, enc_hidden)
        self.mid_linear = nn.Sequential(nn.Linear(enc_hidden, enc_hidden),
                                     nn.ReLU(),
                                     nn.Linear(enc_hidden, dec_hidden),
                                     nn.Tanh())
        self.decoder = nn.LSTM(self.fourier_size, dec_hidden)
        self.out_linear = nn.Sequential(nn.Linear(dec_hidden, output_size),
                                    nn.LogSoftmax(dim=2))
    
    def tokenize(self, input):
        num_tokens = (input.shape[-1] // self.token_size) + 1
        input_tensor = torch.Tensor(input)
        input_tensor = nn.functional.pad(input_tensor, (0, input.shape[-1] % self.token_size))
        tokens = torch.reshape(input_tensor, (self.token_size, num_tokens))
        return tokens

    def forward(self, input):
        tokenized = self.tokenize(input)
        fouriers = torch.view_as_real(self.strideFourier(tokenized))[:,:,0].squeeze(-1)
        drop = self.drop(fouriers)
        enc_output, enc_hidden = self.encoder(drop)
        cell = enc_hidden[1]
        enc_hidden = enc_hidden[0]
        enc_hidden = self.mid_linear(enc_hidden)
        enc_hidden = (enc_hidden, cell)
        current = tokenized[:,0]
        hidden = enc_hidden
        outputs = torch.zeros(tokenized.shape[-1], self.output_size)
        for i in range(tokenized.shape[-1]):
            fouriers = torch.view_as_real(self.strideFourier(tokenized))[:,:,0].squeeze(-1)
            drop = self.drop(fouriers)
            dec_output, hidden = self.decoder(current, hidden, enc_output)
            output = self.out_linear(dec_output)
            outputs[:,i] = output
            current = torch.argmax(output)
        return outputs

