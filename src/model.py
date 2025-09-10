"""
Model.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DiscamModel(nn.Module):
    """
    CNN model.

    Input: (B, 3, H, W) [0, 1]
    Output: (B, 4) [-1, 1]
        up, right, down, left
        Positive means tend toward that edge.
        Negative means tend away from that edge.
    """

    def __init__(self, res: tuple[int, int]):
        super().__init__()

        self.res = res

        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward. Returns logits.

        x: (B, 3, H, W) [0, 1]
        return: (B, 4) [-1, 1]
        """
        x = self.resnet(x)
        x = self.fc(x)
        return x
