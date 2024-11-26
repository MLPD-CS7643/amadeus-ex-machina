import torch.nn as nn


class GriddyModel(nn.Module):
    def __init__(self, **params):
        super().__init__()

        # START PARAMS
        self.n_blocks = params.pop("n_blocks", 3)
        self.block_depth = params.pop("block_depth", 3)
        self.pad = params.pop("pad", 1)
        self.stride = params.pop("stride", 1)
        self.k_conv = params.pop("k_conv", 3)
        self.maxpool = params.pop("maxpool", 3)
        self.dropout = params.pop("dropout", 0.3)
        self.out_channels = params.pop("out_channels", 64)
        # END PARAMS
        
        self.blocks = nn.ModuleList()
        for n in range(self.n_blocks):
            scale = 2**n
            if n == 0:
                in_ch = 3
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
            if n < self.maxpool:
                block.append(nn.MaxPool2d(kernel_size=2,stride=2))
            self.blocks.append(block)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.output = nn.Linear(int((self.out_channels * 2**(self.n_blocks-1))/4), 10)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.dropout > 0:
            x = self.dropout_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.output(x)