"""
Modified 3D resnet model.
"""

import torch
import torch.nn as nn
from torchvision.transforms import v2 as T

from constants import *


class TsModel(nn.Module):
    """
    Model (B, C, T, H, W) -> (B,)

    Implements an attention mechanism:
    The input image is center cropped with a fixed percentage.
    This new image is stacked with the original, and passed together through the model.

    Uses pretrained 3D resnet. Modifications:
    Input:
        Input takes 6 channels (2 stacked images).
    Unfrozen layers:
        Only first layer (conv) and last layer (fc) unfrozen.
    """

    def __init__(self):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)

        # Set first layer.
        #self.model.blocks[0].conv = torch.nn.Conv3d(6, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        # Set fc.
        self.model.blocks[-1].proj = torch.nn.Linear(2048, 1)

        """
        # Freeze layers.
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.blocks[0].conv.parameters():
            param.requires_grad = True
        for param in self.model.blocks[-1].parameters():
            param.requires_grad = True
        """

        #self.resize = T.Resize(VIDEO_RES[::-1])

    def forward(self, x):
        """
        crop_w = int(x.shape[3] * MODEL_ATTN)
        crop_h = int(x.shape[2] * MODEL_ATTN)
        x_crop = x[:, :, :, crop_h:-crop_h, crop_w:-crop_w]
        x_crop = self.resize(x_crop)

        x = torch.cat((x, x_crop), dim=1)
        """

        return self.model(x)
