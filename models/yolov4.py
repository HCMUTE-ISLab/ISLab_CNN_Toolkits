import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False, dropblock=False, sam=False, eca=False, ws=False, coord=False, hard_mish=False, bcn=False, mbn=False):
        super().__init__()

        # PADDING is (ks-1)/2
        padding = (kernel_size - 1) // 2

        modules: ty.List[ty.Union[nn.Module]] = []
        #Adding two more to input channels if coord
        if coord:
            in_channels += 2
            modules.append(AddCoordChannels())
        if ws:
            modules.append(Conv2dWS(in_channels, out_channels, kernel_size, stride, padding, bias=bias))            
        else:
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if bn:
            if bcn:
                modules.append(BCNorm(out_channels, estimate=True))
            elif mbn:
                modules.append(EstBN(out_channels))
            else:
                modules.append(nn.BatchNorm2d(out_channels, track_running_stats=not ws)) #IF WE ARE NOT USING track running stats and using WS, it just explodes.
        if activation == "mish":
            if hard_mish:
                modules.append(HardMish())
            else:
                modules.append(Mish())
        elif activation == "relu":
            modules.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            modules.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            raise BadArguments("Please use one of suggested activations: mish, relu, leaky, linear.")

        if sam:
            modules.append(SAM(out_channels))

        if eca:
            modules.append(ECA())

        if dropblock:
            modules.append(DropBlock2D())

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        y = self.module(x)
        return y
