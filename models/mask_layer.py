import torch
import torch.nn as nn


class mask_layer(nn.Module):
    def __init__(self, size: int) -> None:
        super(mask_layer, self).__init__()
        self.mask = nn.Parameter(torch.ones(size))

    def forward(self, input):
        return input-1+torch.abs(self.mask)
