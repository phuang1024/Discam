"""
Implements models.
"""

import torch
import torch.nn as nn

from constants import *


class TimestampModel(nn.Module):
    """
    Encoder and classifier.

    3D resnet video encoder.
        (B, C, T, H, W) -> (B, D)
    Linear classifier.
        (B, D) -> (B, 1)
    """

    def __init__(self):
        super().__init__()

        self.encoder = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        self.encoder.blocks[-1].proj = nn.Identity()

        # Freeze layers.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.blocks[-1].parameters():
            param.requires_grad = True
        """

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def encode(self, x):
        """Only run encoder."""
        return self.encoder(x)
