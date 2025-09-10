"""
Model.
"""

import torch
import torch.nn as nn
#from torchvision.models import resnet18, ResNet18_Weights


class DiscamModel(nn.Module):
    """
    CNN model.

    Input: (B, 3, H, W) [0, 1]
    Output: (B, 4) [-1, 1]
        up, right, down, left
        Positive means tend toward that edge.
        Negative means tend away from that edge.
    """

    def __init__(self, res: tuple[int, int], output_temp):
        super().__init__()
        self.res = res
        self.output_temp = output_temp

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(4),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(4),
        )

        # For 640x360, this is 40*22.5*16 = 14400
        out_neurons = (res[0] // 16) * (res[1] // 16) * 16
        self.fc = nn.Sequential(
            nn.Linear(out_neurons, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward. Returns logits / temp.

        x: (B, 3, H, W) [0, 1]
        return: (B, 4) [-1, 1]
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x / self.output_temp
        return x
