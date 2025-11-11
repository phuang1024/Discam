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

        num_input_neurons = self.num_hidden_layers * 384 * (res[0] // 14) * (res[1] // 14)
        self.head = nn.Sequential(
            nn.Linear(num_input_neurons, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 4),
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
            x = x.view(x.size(0), -1)

        x = self.head(x)
        x = x / self.output_temp
        return x
