"""
Dataset classes for training.
"""

import json
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from constants import *


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


class SimulatedDataset(Dataset):
    """
    Conventional PyTorch dataset. Loads simulated data from train.py:simulate()

    XY pairs:
        x: frame (3, H, W) float32 tensor in [0, 1] range
        y: expected edge weights (4,) float32 [-1, 1] tensor
            This is computed with the difference between agent's bbox and gt bbox.
    """

    def __init__(self, dirs: list[Path]):
        self.dirs = dirs

        # Lengths of individual dirs.
        self.lengths = []
        for d in dirs:
            length = 0
            for f in d.iterdir():
                if "jpg" in f.suffix:
                    num = int(f.stem.split(".")[0])
                    if num > length:
                        length = num
            self.lengths.append(length)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # Find which dir the index belongs to.
        dir_index = 0
        while index >= self.lengths[dir_index]:
            index -= self.lengths[dir_index]
            dir_index += 1
        dir = self.dirs[dir_index]

        frame_path = dir / f"{index}.frame.jpg"
        agent_path = dir / f"{index}.agent.json"
        gt_path = dir / f"{index}.gt.json"

        frame = read_image(str(frame_path)).float() / 255
        with open(agent_path, "r") as f:
            agent_bbox = torch.tensor(json.load(f), dtype=torch.float32)
        with open(gt_path, "r") as f:
            gt_bbox = torch.tensor(json.load(f), dtype=torch.float32)

        edge_weights = compute_edge_weights(agent_bbox, gt_bbox)

        return frame, edge_weights


def compute_edge_weights(agent_bbox, gt_bbox) -> torch.Tensor:
    """
    Compute edge weights from agent bbox and gt bbox.

    agent_bbox: (4,) tensor, float32
    gt_bbox: (4,) tensor, float32

    return: (4,) tensor, float32
        Edge weights in order: (up, right, down, left)
    """
    edge_weights = torch.tensor([
        agent_bbox[1] - gt_bbox[1],  # up
        gt_bbox[2] - agent_bbox[2],  # right
        gt_bbox[3] - agent_bbox[3],  # down
        agent_bbox[0] - gt_bbox[0],  # left
    ])
    edge_weights = torch.tanh(edge_weights / EDGE_WEIGHT_TEMP)

    return edge_weights
