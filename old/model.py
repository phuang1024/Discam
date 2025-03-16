"""
PyTorch model.
"""

import torch
import torchvision
import torch.nn as nn


class DiscamModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = nn.Linear(512, 3)
        # TODO add sigmoid?

    def forward(self, x):
        return self.resnet(x)
