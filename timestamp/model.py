"""
3D resnet model.
"""

import torch
import torchvision


def create_model():
    model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
    model.blocks[-1].proj = torch.nn.Linear(2048, 1)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.blocks[-1].proj.parameters():
        param.requires_grad = True

    return model
