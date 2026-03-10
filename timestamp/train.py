"""
3D resnet fine tuning script.
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from constants import *
from model import create_model


class VideoDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir

        # Find dataset length.
        self.length = 0
        for file in dir.iterdir():
            if file.suffix == ".mp4":
                self.length = max(self.length, int(file.stem) + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Read video as 3D array.
        path = self.dir / f"{index}.mp4"

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

        # (C, T, H, W)
        x = torch.stack(frames, dim=0).permute(3, 0, 1, 2)

        # Read label.
        label_path = self.dir / f"{index}.label.txt"
        with open(label_path, "r") as fp:
            label = int(fp.read().strip())

        return x, label


def train(args):
    model = create_model().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Trainable parameters: {num_params}")

    dataset = VideoDataset(args.data)
    train_len = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len])
    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 4,
    }
    train_loader = DataLoader(train_data, **loader_args)
    val_loader = DataLoader(val_data, **loader_args)

    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.blocks[-1].proj.parameters(), lr=LR)

    writer = SummaryWriter(args.output / "logs")
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader)
        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.float().to(DEVICE)

            optim.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optim.step()

            pbar.set_description(f"Train epoch {epoch}: loss={loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            pbar = tqdm(val_loader)
            for x, y in pbar:
                x = x.to(DEVICE)
                y = y.float().to(DEVICE)

                pred = model(x).squeeze()
                loss = criterion(pred, y)

                pbar.set_description(f"Val epoch {epoch}: loss={loss.item():.4f}")
                total_loss += loss.item() * x.size(0)
                correct += ((pred > 0) == (y > 0.5)).sum().item()

            avg_loss = total_loss / len(val_data)
            accuracy = correct / len(val_data)
            writer.add_scalar("val/loss", avg_loss, epoch)
            writer.add_scalar("val/accuracy", accuracy, epoch)

        torch.save(model.state_dict(), args.output / f"latest.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)

    train(args)


if __name__ == "__main__":
    main()
