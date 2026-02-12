"""
NN training script.
"""

import sys
sys.path.append("..")

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from tracking import *


class TrackDataset(Dataset):
    """
    Dataset for track classifier.

    X: (T, 5) tensor of trajectory over time.
    Y: Scalar integer class.
    """

    def __init__(self, paths: list[Path]):
        """
        paths: Paths to dirs of distilled data.
            Will sample from all dirs.
        """
        self.paths = paths

        # Dataset length of each path.
        self.lengths = []
        for path in paths:
            length = 0
            for file in path.iterdir():
                if ".label.txt" in file.name:
                    length += 1
            self.lengths.append(length)

        self.cumul_lengths = [0]
        for length in self.lengths:
            self.cumul_lengths.append(self.cumul_lengths[-1] + length)

    def __len__(self):
        return self.cumul_lengths[-1]

    def __getitem__(self, idx):
        # Find corresponding path and file.
        path_idx = 0
        while idx >= self.cumul_lengths[path_idx + 1]:
            path_idx += 1
        file_idx = idx - self.cumul_lengths[path_idx]

        path = self.paths[path_idx]
        track = torch.load(path / f"{file_idx}.pt")
        with open(path / f"{file_idx}.label.txt") as f:
            label = int(f.read().strip())

        return track, label


def forward_loader(model, loader, criterion, desc=""):
    """
    Yields loss for each batch.
    """
    pbar = tqdm(loader)
    for x, y in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred, y)

        pbar.set_description(f"{desc}: loss={loss.item():.4f}")
        yield loss


def train(args):
    model = TrackClassifier().to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    # TODO
    criterion = torch.nn.CrossEntropyLoss()

    dataset = TrackDataset(args.datas)
    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 8,
    }
    train_len = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_data, **loader_args)
    val_loader = DataLoader(val_data, **loader_args)

    writer = SummaryWriter(args.logdir)

    global_step = 0
    for epoch in range(EPOCHS):
        for loss in forward_loader(model, train_loader, criterion, desc=f"Train epoch {epoch}"):
            loss.backward()
            optim.step()
            optim.zero_grad()

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        with torch.no_grad():
            total_loss = 0
            for loss in forward_loader(model, val_loader, criterion, desc=f"Val epoch {epoch}"):
                total_loss += loss.item()
            avg_loss = total_loss / len(val_loader)
            writer.add_scalar("val/loss", avg_loss, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datas", nargs="+", type=Path)
    parser.add_argument("--logdir", type=Path, required=True)
    args = parser.parse_args()


def vis_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("datas", nargs="+", type=Path)
    args = parser.parse_args()

    dataset = TrackDataset(args.datas)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        track, label = dataset[random.randint(0, len(dataset) - 1)]

        # Remove padding.
        last_ind = 0
        for i in range(len(track)):
            if track[i, 0] == 0:
                last_ind = i
                break

        plt.plot(track[:last_ind, 1], track[:last_ind, 2])
        plt.title(f"Label: {label}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #main()
    vis_data()
