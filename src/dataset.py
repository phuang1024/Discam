"""
Dataset of images and labels.
"""

import random
from pathlib import Path

import numpy as np

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

from constants import *
from video_reader import *


class Augmentation(torch.nn.Module):
    """
    Label invariant augmentations
    """

    def __init__(self):
        super().__init__()
        self.rot = T.RandomRotation(3)
        self.crop = T.RandomResizedCrop(256, scale=(0.8, 1.0), antialias=True)
        self.color = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2)

        self.upper_mask = torch.zeros(256, 256)
        for i in range(192):
            self.upper_mask[i] = 1 - i / 192

    def upper_noise(self, x):
        return x + torch.randn_like(x) * self.upper_mask

    def forward(self, x):
        if random.random() < 0.5:
            x = self.rot(x)
        if random.random() < 0.2:
            x = self.crop(x)
        if random.random() < 0.5:
            x[..., :3, :, :] = self.color(x)
        if random.random() < 0.3:
            x = self.upper_noise(x)
        return x


class DiscamDataset(Dataset):
    """
    Modifications to label:

    Label is translation and scale.
    Translation is usually on the order of 1e-3
        thus, we multiply it by a large number.
    Scale is roughly 1 +- 1e-3
        thus, we subtract 1 and multiply by a large number.
    """

    def __init__(self, data_dir, transl_fac=TRANSL_FAC, scale_fac=SCALE_FAC):
        self.data_dir = Path(data_dir)
        self.transl_fac = transl_fac
        self.scale_fac = scale_fac

        self.aug = Augmentation()
        self.length = 0

        for file in self.data_dir.iterdir():
            if file.stem.isdigit():
                self.length = max(self.length, int(file.stem) + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.data_dir / f"{idx}.jpg"
        img = torchvision.io.read_image(str(img_path)).float() / 255

        label_path = self.data_dir / f"{idx}.txt"
        with open(label_path) as f:
            label = f.read().split()
        tx, ty, scale = map(float, label)
        tx *= self.transl_fac
        ty *= self.transl_fac
        scale = (scale - 1) * self.scale_fac

        if random.random() < 0.3:
            img, (tx, ty, scale) = aug_3dtrans(img, (tx, ty, scale))

        img = self.aug(img)

        return img, torch.tensor([tx, ty, scale])


def aug_3dtrans(img, label):
    """
    Crop image off center, adjusting label in the reverse way.
    This incentivizes the network to re-center the view when off.
    """
    width = img.shape[-1] * random.uniform(0.5, 1)
    height = width * img.shape[-2] / img.shape[-1]
    width, height = int(width), int(height)

    x = random.randint(0, img.shape[-1] - width)
    y = random.randint(0, img.shape[-2] - height)

    dx = np.interp(x, [0, img.shape[-1] - width], [-1, 1])
    dy = np.interp(y, [0, img.shape[-2] - height], [-1, 1])
    scale_fac = width / img.shape[-1]

    new_img = img[..., y:y + height, x:x + width]
    new_label = (
        label[0] - dx * 0.3,
        label[1] - dy * 0.3,
        label[2] + scale_fac * 0.1,
    )

    return new_img, new_label
