import torch
from torch import nn
from models.StridedFourier import StridedFourier

class FourierS2S(nn.Module):

    def __init__(self, kernel_size, stride, fourier_size, enc_hidden, dec_hidden, output_size, token_size, device, dropout=0.2, sample_rate=44100):
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
            device (str): device for torch objects
        """

        super().__init__()
        self.token_size = token_size
        self.fourier_size = fourier_size
        self.output_size = output_size
        self.dec_hidden = dec_hidden
        self.device = device
        self.strideFourier = StridedFourier(kernel_size, stride, fourier_size)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.LSTM(self.fourier_size, enc_hidden, batch_first=True, device=self.device)
        self.mid_linear = nn.Sequential(nn.Linear(enc_hidden, enc_hidden, device=self.device),
                                     nn.ReLU(),
                                     nn.Linear(enc_hidden, dec_hidden, device=self.device),
                                     nn.Tanh())
        self.decoder = nn.LSTM(self.fourier_size, self.dec_hidden,device=self.device)
        #self.out_linear = nn.Linear(dec_hidden, output_size)
        self.out_linear = nn.Sequential(nn.Linear(dec_hidden, output_size, device=self.device),
                                    nn.LogSoftmax(dim=0))
        
    def forward(self, input):
        """
        forward method for FourierS2S
        Args:
            input torch.Tensor((token_size), (token_size)...): chunks of wav data of size token_size in a tensor of size seq_len
        """
        #tokenize, then pass data into encoder, then decoder with "embeddings" as the strided fourier results
        tokenized = input.squeeze(-1)
        fouriers = self.strideFourier(tokenized).real
        drop = self.drop(fouriers)
        enc_output, enc_hidden = self.encoder(drop)
        cell = enc_hidden[1]
        enc_hidden = enc_hidden[0]
        enc_hidden = self.mid_linear(enc_hidden)
        enc_hidden = (enc_hidden, cell)
        hidden = (enc_hidden[0][:,0,:], enc_hidden[0][:,0,:])
        outputs = torch.zeros(tokenized.shape[0], self.output_size).to(self.device)
        for i in range(tokenized.shape[0]):
            current = tokenized[i,:]
            fouriers = self.strideFourier(current.unsqueeze(0)).real
            drop = self.drop(fouriers).squeeze(0)
            dec_output, hidden = self.decoder(drop, hidden)
            output = self.out_linear(dec_output[-1,:])
            outputs[i,:] = output
        return outputs