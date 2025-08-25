"""
CNN model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 1/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 1/4
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
        )
        out_neurons = (res[0] // 4) * (res[1] // 4) * 16
        self.head = nn.Sequential(
            nn.Linear(out_neurons, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x = F.tanh(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
