"""
Dataset classes for training.

Running this file will show a visualization of augmentations.
"""

import json
import random
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from torchvision.io import read_image

from constants import *


class VideosDataset:
    """
    Load frames and bboxes of video.

    Note: This is not a traditional PyTorch dataset.
    It loads chunks of frames randomly.
    """

    def __init__(self, dir: Path, offset: int):
        """
        dir: Directory containing subdirectories of processed video data.
        offset: Number of frames to offset bbox relative to image.
            Nonnegative number.
            Higher value means a frame gets a future bbox.
        """
        self.dir = dir
        self.offset = offset

        self.videos = {}
        for video_dir in dir.iterdir():
            if video_dir.is_dir():
                max_num = 0
                for f in video_dir.iterdir():
                    if f.suffix == ".jpg":
                        num = int(f.stem.split(".")[0])
                        if num > max_num:
                            max_num = num

                self.videos[video_dir.name] = max_num + 1 - offset

    def get_rand_chunk(self, size, step) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a continuous chunk of frames and corresponding bboxes.

        size: Length (i.e. number of frames) of chunk.
        step: Step size between frames in chunk.
        res: (width, height) of frames to load.

        return: (frames, bboxes)
            frames: Uncropped, unresized frames from the video.
                tensor float32 (size, C, H, W) [0, 1]
            bboxes: (x1, y1, x2, y2) corresponding to each frame.
                tensor int64 (size, 4)
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
        i = start
        while len(frames) < size:
            frame_path = self.dir / video_name / f"{i}.frame.jpg"
            frame = read_image(frame_path).float() / 255

            bbox_path = self.dir / video_name / f"{i + self.offset}.allbbox.json"
            with open(bbox_path, "r") as f:
                bbox = json.load(f)
            bbox = torch.tensor(bbox, dtype=torch.int64)

            frames.append(frame)
            bboxes.append(bbox)

            i += step

        frames = torch.stack(frames, dim=0)
        bboxes = torch.stack(bboxes, dim=0)

        return frames, bboxes


class SimulatedDataset(Dataset):
    """
    Conventional PyTorch dataset.
    Loads simulated data generated from train.py:simulate().

    XY pairs:
        x: Image frame.
            tensor float32 (3, H, W) [0, 1]
        y: Ground truth edge weights. This is computed using the difference between agent's bbox and gt bbox.
            tensor float32 (4,) [-1, 1]
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

        # Augmentations
        self.image_augs = T.Compose([
            T.RandomRotation(degrees=10),
            T.ElasticTransform(alpha=10),
            T.RandomResizedCrop(MODEL_INPUT_RES[::-1], scale=(0.8, 1.0), ratio=(0.9, 1 / 0.9)),
            T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.3),
        ])

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
        frame = read_image(str(frame_path)).float() / 255
        if random.random() < AUG_FREQ:
            frame = self.image_augs(frame)

        agent_path = dir / f"{index}.agent.json"
        gt_path = dir / f"{index}.gt.json"
        with open(agent_path, "r") as f:
            agent_bbox = torch.tensor(json.load(f), dtype=torch.float32)
        with open(gt_path, "r") as f:
            gt_bbox = torch.tensor(json.load(f), dtype=torch.float32)
        edge_weights = compute_edge_weights(agent_bbox, gt_bbox)

        return frame, edge_weights


def compute_edge_weights(agent_bbox, gt_bbox) -> torch.Tensor:
    """
    Compute ground truth edge weights from the difference between agent and gt bbox.

    agent_bbox:
        tensor float32 (4,)
    gt_bbox:
        tensor float32 (4,)

    return: Edge weights in order: (up, right, down, left)
        tensor float32 [-1, 1]
    """
    edge_weights = torch.tensor([
        agent_bbox[1] - gt_bbox[1],  # up
        gt_bbox[2] - agent_bbox[2],  # right
        gt_bbox[3] - agent_bbox[3],  # down
        agent_bbox[0] - gt_bbox[0],  # left
    ])
    edge_weights = torch.tanh(edge_weights / EDGE_WEIGHT_TEMP)

    return edge_weights


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="NOTE: Should be a simulated epoch from training.")
    args = parser.parse_args()

    dataset = SimulatedDataset([args.data])

    to_pil = T.ToPILImage()
    for i in range(15):
        img, _ = dataset[i]
        img = to_pil(img)

        plt.subplot(5, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("augs.png", dpi=300)
