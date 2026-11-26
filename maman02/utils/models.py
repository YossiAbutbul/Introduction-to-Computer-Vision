"""
models.py - my CNN (trained from scratch).
"""

import torch.nn as nn


def conv_bn(c_in, c_out, stride=1):
    """3x3 conv + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    """Two 3x3 convs with a skip connection: out = relu(x + F(x)).
    With stride 2 the skip path gets a 1x1 conv so the sizes match."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = conv_bn(c_in, c_out, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out))
        self.skip = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.skip(x) + self.conv2(self.conv1(x)))


class PetCNN(nn.Module):
    """Small ResNet-style CNN, ~2.8M parameters.
    224x224x3 -> stem -> 56x56x32 -> 4 stages -> 7x7x256 -> avg pool -> linear.
    n_out = 1 gives one logit (dog = positive)."""

    def __init__(self, n_out=1):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn(3, 32, stride=2),              # 112x112
            conv_bn(32, 32),
            nn.MaxPool2d(3, stride=2, padding=1))  # 56x56
        self.stages = nn.Sequential(
            ResBlock(32, 32), ResBlock(32, 32),                # 56x56x32
            ResBlock(32, 64, stride=2), ResBlock(64, 64),      # 28x28x64
            ResBlock(64, 128, stride=2), ResBlock(128, 128),   # 14x14x128
            ResBlock(128, 256, stride=2), ResBlock(256, 256))  # 7x7x256
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),  # 256
            nn.Dropout(0.3),
            nn.Linear(256, n_out))
        self.n_out = n_out

    def forward(self, x):
        out = self.head(self.stages(self.stem(x)))
        return out.squeeze(1) if self.n_out == 1 else out
