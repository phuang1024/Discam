"""
Dataset for training.
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from constants import *


class VideoDataset(Dataset):
    """
    x: Video clip as (C, T, H, W) tensor.
    y: Label as int.
    """

    def __init__(self, dirs):
        self.dirs = list(dirs)

        # Find dataset lengths.
        self.lengths = []
        for dir in self.dirs:
            length = 0
            for file in dir.iterdir():
                if file.suffix == ".mp4":
                    length = max(length, int(file.stem) + 1)
            self.lengths.append(length)

        # Augmentation.
        aspect = VIDEO_RES[0] / VIDEO_RES[1]
        self.aug = T.Compose([
            T.RandomResizedCrop(VIDEO_RES[::-1], (0.5, 1), (aspect, aspect)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.1),
            #T.GaussianNoise(sigma=0.02),
        ])

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        # Find dir index.
        dir_index = 0
        while index >= self.lengths[dir_index]:
            index -= self.lengths[dir_index]
            dir_index += 1

        # Read video as 3D array.
        path = self.dirs[dir_index] / f"{index}.mp4"

        video = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = torch.from_numpy(frame).float() / 255
            frames.append(frame)
        video.release()

        # Pad to 16 frames if necessary.
        while len(frames) < 16:
            frames.append(frames[-1])

        # (T, H, W, C)
        x = torch.stack(frames, dim=0)
        # (T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.aug(x)
        # (C, T, H, W)
        x = x.permute(1, 0, 2, 3)

        # Read label.
        label_path = self.dirs[dir_index] / f"{index}.label.txt"
        with open(label_path, "r") as fp:
            label = int(fp.read().strip())

        return x, label
