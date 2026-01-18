"""
Video data reading,
and Torch dataset.
"""

import json
from pathlib import Path

from torchvision.io import read_image


class VideoReader:
    """
    Reader and indexer for frames and bboxes of a video.

    The video directory contains frames and corresponding bboxes.
    Each frame is labeled `0.frame.jpg`, and bbox `0.allbbox.json`.
    Obviously, `0` can be changed to be any non-negative integer.
    Indexing begins at 0.
    """

    def __init__(self, dir: Path):
        """
        dir: Path to frames and bboxes directory.
        """
        self.dir = dir

        self.length = 0

        for file in dir.iterdir():
            num = file.name.split(".")[0]
            if num.isdigit():
                self.length = max(self.length, int(num) + 1)

    def read(self, index: int):
        """
        return: (frame, bbox)
            frame: Original resolution image.
                tensor int [0-255] (C, H, W)
            bbox: (x1, y1, x2, y2)
        """
        frame_path = self.dir / f"{index}.frame.jpg"
        bbox_path = self.dir / f"{index}.allbbox.json"

        frame = read_image(str(frame_path))

        with open(bbox_path, "r") as f:
            bbox = json.load(f)

        return frame, bbox


class DiscamDataset(Dataset):
    """
    Torch dataset subclass for loading consecutive frames and bboxes.
    """

    def __init__(self, video_dir: Path):
        """
        video_dir: Directory with video subdirectories.
        """
        self.video_dir = video_dir
        self.video_readers = []
        for video_subdir in video_dir.iterdir():
            if video_subdir.is_dir():
                self.video_readers.append(VideoReader(video_subdir))

        self.cumulative_lengths = []
        cum_length = 0
        for reader in self.video_readers:
            cum_length += reader.length
            self.cumulative_lengths.append(cum_length)
