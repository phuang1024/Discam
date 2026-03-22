"""
Video clip dataset.
"""

import cv2

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from constants import *


class VideoDataset(Dataset):
    """
    Video clip dataset.

    Return data:
        x: Video clip as (C, T, H, W) tensor.
        y: Label as int.
    """

    def __init__(self, dir):
        self.dir = dir

        # List of (video_path, label_data, frame_count)
        self.videos = []
        for file in self.dir.iterdir():
            if file.suffix == ".mp4":
                # Check for label.
                label_path = self.dir / f"{file.stem}.txt"
                if not label_path.exists():
                    continue

                # Read label data.
                label = read_ts(label_path)

                # Get frame count.
                video = cv2.VideoCapture(str(file))
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                video.release()

                length = frame_count // VIDEO_LEN
                if length > 0:
                    self.videos.append((file, label, length))

        # Augmentation.
        aspect = VIDEO_RES[0] / VIDEO_RES[1]
        self.aug = T.Compose([
            T.RandomResizedCrop(VIDEO_RES[::-1], (0.5, 1), (aspect, aspect)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.1),
            #T.GaussianNoise(sigma=0.02),
        ])

    def __len__(self):
        return sum(x[2] for x in self.videos)

    def __getitem__(self, index):
        # Find dir index.
        vid_index = 0
        while index >= self.videos[vid_index][2]:
            index -= self.videos[vid_index][2]
            vid_index += 1

        # Read video as 3D array.
        path = self.videos[vid_index][0]
        video = cv2.VideoCapture(str(path))
        video.set(cv2.CAP_PROP_POS_FRAMES, index * VIDEO_LEN)
        frames = []
        for _ in range(VIDEO_LEN):
            ret, frame = video.read()
            if not ret:
                raise RuntimeError(f"Failed to read sample {index} from video {path}")
            frame = torch.from_numpy(frame).float() / 255
            frames.append(frame)
        video.release()

        # (T, H, W, C)
        x = torch.stack(frames, dim=0)
        # (T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.aug(x)
        # (C, T, H, W)
        x = x.permute(1, 0, 2, 3)

        # Find label.
        frame = index * VIDEO_LEN
        label = 0
        for start, end in self.videos[vid_index][1]:
            if start <= frame < end:
                label = 1
                break

        return x, label


def read_ts(path) -> list[tuple[float, float]]:
    """
    Read timestamps from file.
    """
    ret = []
    with open(path, "r") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            start = parse_time(parts[0])
            end = parse_time(parts[-1])
            ret.append((start, end))

    return ret


def parse_time(string):
    """
    Convert time string to seconds.
    """
    parts = string.split(":")
    s = 0
    m = 0
    h = 0
    if len(parts) == 1:
        s = float(parts[0])
    elif len(parts) == 2:
        s = float(parts[1])
        m = float(parts[0])
    elif len(parts) == 3:
        s = float(parts[2])
        m = float(parts[1])
        h = float(parts[0])
    return s + 60*m + 3600*h
