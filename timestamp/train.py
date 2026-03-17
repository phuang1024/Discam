"""
3D resnet fine tuning script.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from constants import *
from dataset import VideoDataset
from model import TCNModel


def train(args):
    model = TCNModel().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Trainable parameters: {num_params}")

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))

    dataset = VideoDataset(args.data)
    train_len = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len])
    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 2,
    }
    train_loader = DataLoader(train_data, **loader_args)
    val_loader = DataLoader(val_data, **loader_args)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT).to(DEVICE))
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


if __name__ == "__main__":
    main()
