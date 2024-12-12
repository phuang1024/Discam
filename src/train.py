"""
Train the model.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from dataset import DiscamDataset
from model import DiscamModel


def train(args):
    dataset = DiscamDataset(args.data)
    train_size = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
    loader_args = {"batch_size": args.batch_size, "shuffle": True}
    train_loader = DataLoader(train_data, **loader_args)
    val_loader = DataLoader(val_data, **loader_args)

    model = DiscamModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        for img, label in (pbar := tqdm(train_loader)):
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Train epoch {epoch}: Loss: {loss.item():.4f}")

        with torch.no_grad():
            for img, label in (pbar := tqdm(val_loader)):
                pred = model(img)
                loss = criterion(pred, label)

                pbar.set_description(f"Val epoch {epoch}: Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), "model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
