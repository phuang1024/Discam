"""
Model.
"""

import torch
import torch.nn as nn

from constants import *


class DiscamModel(nn.Module):
    """
    CNN and fully connected model.

    Input: (B, 3*N, H, W) [0, 1]
        N previous frames (for recurrence),
            with the current frame being the first three channels (0 to 2).
    Output: (B, 4) [-1, 1]
        (up, right, down, left)
        Positive means tend toward that edge. Negative means tend away from that edge.
            I.e. for the bottom edge, positive means move that edge down.
        Note: Output is logits / temp, so can be outside [-1, 1].
    """

    def __init__(self, res: tuple[int, int], output_temp):
        super().__init__()
        self.res = res
        self.output_temp = output_temp

        self.conv = nn.Sequential(
            nn.Conv2d(3 * RNN_FRAMES, 8, 3, padding=1),
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

        # For 640x360, this is 40*22*16 = 14080
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
