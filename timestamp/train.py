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
from torchvision.transforms import v2 as T

from constants import *
from model import TsModel


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


def train(args):
    model = TsModel().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Trainable parameters: {num_params}")

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))

    dataset = VideoDataset(args.data.iterdir())
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
    optim = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LR)

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
    parser.add_argument("--resume")
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)

    train(args)


def vis_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()

    dataset = VideoDataset(args.data)

    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"Label: {y}")
        for t in range(x.size(1)):
            frame = (x[:, t] * 255).byte().permute(1, 2, 0).numpy()
            cv2.imshow("frame", frame)
            cv2.waitKey(200)


if __name__ == "__main__":
    main()
    #vis_data()
