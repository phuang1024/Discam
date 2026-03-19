"""
Modified 3D resnet model.
"""

import torch
import torch.nn as nn

from constants import *


class TsModel(nn.Module):
    """
    Model (B, C, T, H, W) -> (B,)
    """

    def __init__(self):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        # Set fc.
        self.model.blocks[-1].proj = torch.nn.Linear(2048, 1)

        # Freeze layers.
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.blocks[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
