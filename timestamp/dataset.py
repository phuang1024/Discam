"""
Dataset for training.
"""

import argparse
from pathlib import Path

import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from constants import *


class VideoDataset(Dataset):
    """
    Dataset of multiple video files.
    Samples intervals of videos.

    Note: This class does not perform any frame resizing or stepping.
    Make sure you do those pre-processing, e.g. with ffmpeg.

    x: Video clip as (T, C, H, W) tensor.
    y: Label per frame as (T,) tensor.
    """

    def __init__(self, dir):
        """
        dir: Directory with video and label files.
        """
        self.dir = Path(dir)

        # List of (video_path, label_data, length).
        self.videos = []

        for file in self.dir.iterdir():
            if file.suffix == ".mp4":
                label_path = file.with_suffix(".txt")
                if not label_path.exists():
                    print("VideoDataset: No label for video", file)
                    continue

                cap = cv2.VideoCapture(str(file))
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                length = num_frames - VIDEO_LEN + 1
                cap.release()

                label_data = read_label_file(label_path, num_frames)

                self.videos.append((file, label_data, length))

    def __len__(self):
        return sum(x[2] for x in self.videos)

    def __getitem__(self, index):
        # Find dir index.
        dir_index = 0
        while index >= self.videos[dir_index][2]:
            index -= self.videos[dir_index][2]
            dir_index += 1

        # Read video as 3D array.
        path = self.videos[dir_index][0]
        x = self.read_video_chunk(path, index)

        # Get per frame labels.
        label = self.videos[dir_index][1][index : index+VIDEO_LEN]

        assert x.shape[0] == VIDEO_LEN
        assert label.shape[0] == VIDEO_LEN
        return x, label

    def read_video_chunk(self, path, start):
        """
        Read a chunk of video starting at frame start.
        Returns (T, C, H, W) tensor.
        """
        video = cv2.VideoCapture(str(path))
        video.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(VIDEO_LEN):
            ret, frame = video.read()
            if not ret:
                break
            frame = torch.from_numpy(frame).float() / 255
            frames.append(frame)
        video.release()

        # (T, H, W, C)
        x = torch.stack(frames, dim=0)
        # (T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


def read_label_file(path, length) -> torch.Tensor:
    """
    Called at dataset initialization to read label file.

    Returns tensor of shape (length,) dtype float.
    The values around each "point start" event are 1, and the rest are 0.
    Determined by POS_LABEL_RADIUS.
    """
    events = []
    with open(path, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            start = line.split()[0]
            events.append(parse_time(start))

    label = torch.zeros(length)
    for event in events:
        start = int(event - POS_LABEL_RADIUS)
        end = int(event + POS_LABEL_RADIUS) + 1
        start = max(start, 0)
        end = min(end, length)
        label[start:end] = 1

    return label


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


def vis_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    dataset = VideoDataset(args.data)

    total = 0
    ones = 0
    for x, y in dataset:
        total += y.size(0)
        ones += y.sum().item()
    print(f"total={total}, positive={ones}, ratio={ones/total:.4f}")

    exit()

    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"Label: {y}")
        for t in range(x.size(1)):
            frame = (x[:, t] * 255).byte().permute(1, 2, 0).numpy()
            cv2.imshow("frame", frame)
            cv2.waitKey(200)


if __name__ == "__main__":
    vis_data()
