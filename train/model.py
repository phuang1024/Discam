"""
Model.
"""

import torch
import torch.nn as nn

from constants import *

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
DINO = DINO.to(DEVICE)
DINO.eval()


class DiscamModel(nn.Module):
    """
    CNN and fully connected model.

    Input: (B, 3, H, W) [0, 1]
    Output: (B, 4) [-1, 1]
        (up, right, down, left)
        Positive means tend toward that edge. Negative means tend away from that edge.
            I.e. for the bottom edge, positive means move that edge down.
        Note: Output is logits / temp, so can be outside [-1, 1].
    """

    def __init__(
            self,
            res: tuple[int, int] = MODEL_INPUT_RES,
            output_temp = EDGE_WEIGHT_TEMP,
            num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.res = res
        self.output_temp = output_temp
        self.num_hidden_layers = num_hidden_layers

        # Input is (B, 384, H/14, W/14)
        # Output is (B, 64, H/14/4, W/14/4)
        self.conv = nn.Sequential(
            nn.Conv2d(384 * self.num_hidden_layers, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_bypass = nn.Conv2d(384 * self.num_hidden_layers, 64, 1, 4, 0)

        num_input_neurons = 64 * (res[0] // 14 // 4) * (res[1] // 14 // 4)
        self.head = nn.Sequential(
            nn.Linear(num_input_neurons, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward. Returns logits / temp.

        x: (B, 3, H, W) [0, 1]
        return: (B, 4) [-1, 1]
        """
        with torch.no_grad():
            x = DINO.get_intermediate_layers(x, n=self.num_hidden_layers, reshape=True)
            x = torch.cat(x, dim=1)

        x = self.conv(x) + self.conv_bypass(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x = x / self.output_temp
        return x
