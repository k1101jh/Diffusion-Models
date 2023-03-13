import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        pass


class Upsample(nn.Module):
    def __init__(self, sampling_mode='bilinear'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=sampling_mode, align_corners=True)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = nn.GroupNorm()

    def forward(self, x):
        return x


class Up(nn.Module):
    def __init__(self):
        super().__init__()


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, noise_steps=1000, time_dim=256, features=None):
        super().__init__()

    def forward(self, x):
        return x