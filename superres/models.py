"""Original compact residual and EDSR-style pixel-shuffle implementations."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, features, residual_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
        )
        self.residual_scale = residual_scale

    def forward(self, x):
        return x + self.body(x) * self.residual_scale


class EDSR(nn.Module):
    def __init__(self, scale=2, features=48, blocks=8, residual_scale=0.1):
        super().__init__()
        if scale not in (2, 4) or features < 1 or blocks < 1:
            raise ValueError("Invalid scale/features/blocks")
        self.scale, self.blocks = scale, blocks
        self.register_buffer("mean", torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))
        self.head = nn.Conv2d(3, features, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(features, residual_scale) for _ in range(blocks)],
            nn.Conv2d(features, features, 3, padding=1),
        )
        up = []
        for _ in range(int(math.log2(scale))):
            up += [nn.Conv2d(features, features * 4, 3, padding=1), nn.PixelShuffle(2)]
        self.tail = nn.Sequential(*up, nn.Conv2d(features, 3, 3, padding=1))
        # Conservative LR halo covers every convolution's receptive field.
        self.required_halo = 2 * blocks + 6

    def forward(self, x):
        x = self.head(x - self.mean)
        return self.tail(x + self.body(x)) + self.mean


class CompactSR(nn.Module):
    def __init__(self, scale=2, features=32, blocks=4):
        super().__init__()
        if scale not in (2, 4) or features < 1 or blocks < 1:
            raise ValueError("Invalid scale/features/blocks")
        self.scale = scale
        self.head = nn.Conv2d(3, features, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(features, 1.0) for _ in range(blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(features, 3 * scale**2, 3, padding=1), nn.PixelShuffle(scale)
        )
        self.required_halo = 2 * blocks + 4

    def forward(self, x):
        baseline = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return baseline + self.tail(self.body(self.head(x)))


class Bicubic(nn.Module):
    def __init__(self, scale):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4")
        self.scale, self.required_halo = scale, 3

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


def build_model(cfg):
    name = cfg["architecture"]
    if name == "bicubic":
        return Bicubic(cfg["scale"])
    if name == "compact":
        return CompactSR(cfg["scale"], cfg["features"], cfg["blocks"])
    if name == "edsr":
        return EDSR(cfg["scale"], cfg["features"], cfg["blocks"], cfg.get("residual_scale", 0.1))
    raise ValueError("architecture must be compact, edsr or bicubic")
