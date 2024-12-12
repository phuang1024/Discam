"""
Dataset of images and labels.
"""

from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset

from constants import *
from video_reader import *


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

        return img, torch.tensor([tx, ty, scale])
