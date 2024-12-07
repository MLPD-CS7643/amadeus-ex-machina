import torch
from torch import nn
from models import StridedFourier

class FourierS2S(nn.Module):

    def __init__(self, kernel_size, stride, fourier_size, enc_hidden, dec_hidden, output_size, token_size, dropout=0.2, sample_rate=44100):
        """
        Initialize model
        Args:
            kernel_size (int): size of kernel to stride across token (must be smaller than token size)
            stride (int): stride length for the kernel across the token
            fourier_size (int): generally going to be the same as kernel size
            enc_hidden (int): size of the hidden state of the encoder
            dec_hidden (int): size of the hidden state of the decoder
            output_size (int): number of classes
            token_size (int): number of bits in a token
            dropout (float): dropout percentage
            sample_rate (int): number of bits/sec from the wav file (may not be neccesary)
        """

        super().__init__()
        self.token_size = token_size
        self.fourier_size = fourier_size
        self.output_size = output_size
        self.dec_hidden = (self.token_size // fourier_size) * self.token_size
        self.strideFourier = StridedFourier(kernel_size, stride, fourier_size)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.LSTM(self.fourier_size, enc_hidden)
        self.mid_linear = nn.Sequential(nn.Linear(enc_hidden, enc_hidden),
                                     nn.ReLU(),
                                     nn.Linear(enc_hidden, self.token_size),
                                     nn.Tanh())
        self.decoder = nn.LSTM(self.fourier_size, self.dec_hidden)
        self.out_linear = nn.Sequential(nn.Linear(dec_hidden, output_size),
                                    nn.LogSoftmax(dim=2))
        
    def forward(self, input):
        #tokenize, then pass data into encoder, then decoder with "embeddings" as the strided fourier results
        tokenized = self.tokenize(input)
        fouriers = self.strideFourier(tokenized.transpose(0,1)).real
        drop = self.drop(fouriers)
        enc_output, enc_hidden = self.encoder(drop)
        cell = enc_hidden[1]
        enc_hidden = enc_hidden[0]
        enc_hidden = self.mid_linear(enc_hidden)
        enc_hidden = (enc_hidden, cell)
        current = tokenized.transpose(0,1)[0,:]
        hidden = enc_hidden
        outputs = torch.zeros(tokenized.shape[-1], self.output_size)
        for i in range(tokenized.shape[-1]):
            fouriers = self.strideFourier(current.unsqueeze(0)).real
            drop = self.drop(fouriers).squeeze(0)
            #TODO Fix bugs occuring related to hidden size
            reshaped_hidden = (torch.reshape(hidden[0], (1,-1)), torch.reshape(hidden[1], (1,-1)))
            dec_output, hidden = self.decoder(drop, reshaped_hidden)
            output = self.out_linear(dec_output)
            outputs[:,i] = output
            current = torch.argmax(output)
        return outputs