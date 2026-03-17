"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import v2 as T

from constants import *


class ResTCNBlock(nn.Module):
    """
    1D dilated CNN and residual connection.
    """

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            in_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x + res
        return x


class TCNModel(nn.Module):
    """
    Model (B, T, C, H, W) -> (B, T)

    ResNet independently encodes each frame (T dimension).
    Embeddings passed through 1D CNN.
    """

    def __init__(self):
        super().__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove head.
        self.resnet.fc = nn.Identity()
        # Freeze.
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.tcn = nn.Sequential(
            # Projection.
            nn.Conv1d(2048, 256, 1),

            ResTCNBlock(256, 256, dilation=1),
            ResTCNBlock(256, 256, dilation=2),
            ResTCNBlock(256, 256, dilation=4),

            # Classifier.
            nn.Conv1d(256, 1, 1),
        )

    def forward(self, x):
        embeds = []
        for t in range(x.shape[1]):
            frame = x[:, t]  # (B, C, H, W)
            em = self.resnet(frame)  # (B, 2048)
            embeds.append(em)
        embeds = torch.stack(embeds, dim=1)  # (B, T, 2048)

        embeds = embeds.permute(0, 2, 1)  # (B, 2048, T)
        out = self.tcn(embeds)  # (B, 1, T)
        out = out.squeeze(1)  # (B, T)
        return out
