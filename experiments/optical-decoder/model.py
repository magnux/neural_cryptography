"""Small untrained optical symbol classifier used for M1 browser feasibility.

The network is deliberately independent of cryptographic keys and wire parsing.
Its input is one calibrated, normalized RGB payload crop in NCHW layout; its
output is a physical-palette logit for each of the 16 classes at every grid cell.
"""

import torch
from torch import nn
from torch.nn import functional as F


def conv(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True),
        nn.ReLU(),
    )


class OpticalDecoder(nn.Module):
    """121,792-parameter, single-frame, four-raster-pixels-per-cell U-Net.

    The exported M1 profiles are batch-one G=64 and G=128. There is no softmax
    here: the receiver masks inactive palette classes before normalizing logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inlet = nn.Sequential(conv(3, 8, 2), conv(8, 16, 2))
        self.e0 = nn.Sequential(conv(16, 16), conv(16, 16))
        self.e1 = nn.Sequential(conv(16, 32, 2), conv(32, 32))
        self.bottleneck = nn.Sequential(conv(32, 64, 2), conv(64, 64))
        self.d1 = nn.Sequential(conv(96, 32), conv(32, 32))
        self.d0 = nn.Sequential(conv(48, 16), conv(16, 16))
        self.head = nn.Conv2d(16, 16, 1, bias=True)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        e0 = self.e0(self.inlet(pixels))
        e1 = self.e1(e0)
        context = self.bottleneck(e1)
        d1 = self.d1(torch.cat((F.interpolate(context, scale_factor=2, mode="nearest"), e1), dim=1))
        d0 = self.d0(torch.cat((F.interpolate(d1, scale_factor=2, mode="nearest"), e0), dim=1))
        return self.head(d0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
