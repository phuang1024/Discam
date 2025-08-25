"""
"""

import json
import random
from pathlib import Path

import cv2
import torch
from torchvision.io import read_image


class VideosDataset:
    """
    Load frames and bboxes of video.

    Note: This is not a traditional PyTorch dataset.
    It is a handler that loads chunks of frames randomly.
    """

    def __init__(self, dir: Path):
        self.dir = dir

        self.videos = {}
        for video_dir in dir.iterdir():
            if video_dir.is_dir():
                max_num = 0
                for f in video_dir.iterdir():
                    if f.suffix == ".jpg":
                        num = int(f.stem.split(".")[0])
                        if num > max_num:
                            max_num = num

                self.videos[video_dir.name] = max_num + 1

    def get_rand_chunk(self, size, step, res) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a continuous chunk of frames and corresponding bboxes, length = `size`.

        size: Length of chunk.
        step: Step size between frames in chunk.
        res: (width, height) of frames to load.

        return: (frames, bboxes)
            frames: (size, C, H, W) tensor, float32, in [0, 1] range
            bboxes: (size, 4) tensor, int64
                (x1, y1, x2, y2) format
        """
        # Sample random video weighted by length.
        video_name = random.choices(
            list(self.videos.keys()),
            weights=list(self.videos.values())
        )[0]
        video_len = self.videos[video_name]

        # Sample random start frame.
        start = random.randint(0, video_len - size * step)

        # Read data.
        frames = []
        bboxes = []
        for i in range(start, start + size * step, step):
            frame_path = self.dir / video_name / f"{i}.frame.jpg"
            bbox_path = self.dir / video_name / f"{i}.allbbox.json"

            frame = cv2.imread(str(frame_path))
            frame = cv2.resize(frame, res)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255

            with open(bbox_path, "r") as f:
                bbox = json.load(f)
            bbox = torch.tensor(bbox, dtype=torch.int64)

            frames.append(frame)
            bboxes.append(bbox)

        frames = torch.stack(frames, dim=0)
        bboxes = torch.stack(bboxes, dim=0)

        return frames, bboxes
