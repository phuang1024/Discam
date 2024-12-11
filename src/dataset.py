"""
Handles processing training data.
Includes extracting video frames and running the transform solver.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from video_reader import *


class DiscamDataset(Dataset):
    def __init__(self, data_dir, keyframe: int):
        self.data_dir = Path(data_dir)
        self.keyframe = keyframe

        # Map of file name to number of samples available.
        # Number of samples is num_frames - keyframe_interval.
        self.samples = {}
        for file in self.data_dir.iterdir():
            length = get_video_length(str(file))
            self.samples[file] = length - keyframe

    def __len__(self):
        return sum(self.samples.values())

    def __getitem__(self, idx):
        # Convert to file name and sample index.
        for file, num_samples in self.samples.items():
            if idx < num_samples:
                break
            idx -= num_samples
        else:
            raise IndexError("Invalid index")

