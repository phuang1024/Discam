"""
Main training script.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
from tqdm import trange

from agent import Agent
from constants import *
from dataset import VideosDataset, SimulatedDataset
from model import DiscamModel


@torch.no_grad()
def simulate(videos_dataset, agent, data_dir: Path):
    """
    Simulate data for one epoch of training.

    videos_dataset: Instance of VideosDataset to sample from.
    agent: Agent to use for simulation.
    data_dir: Where to save data.
    """
    # Resize transform for model input (H, W).
    resize = Resize(MODEL_INPUT_RES[::-1])

    data_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    for sim in trange(SIMS_PER_EPOCH, desc="Simulating"):
        # Reset environment.
        frames, bboxes = videos_dataset.get_rand_chunk(SIM_STEPS, SIM_FRAME_SKIP)
        frames = frames.to(DEVICE)

        # Set agent bbox to first gt bbox.
        #randx, randy = (torch.randn((2,)) * SIM_START_RANDOM).tolist()
        agent.bbox = bboxes[0].tolist()
        """
        agent.bbox[0] += randx
        agent.bbox[2] += randx
        agent.bbox[1] += randy
        agent.bbox[3] += randy
        """

        for step in range(SIM_STEPS):
            frame = frames[step]

            # Step agent.
            bbox = list(map(int, agent.bbox))
            frame = frame[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
            frame = resize(frame.unsqueeze(0))[0]
            agent.step(frame)

            # Save frame and bbox.
            frame = frame.cpu().permute(1, 2, 0).numpy() * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(data_dir / f"{index}.frame.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            with open(data_dir / f"{index}.agent.json", "w") as f:
                json.dump([int(x) for x in agent.bbox], f)

            gt_bbox = bboxes[step]
            with open(data_dir / f"{index}.gt.json", "w") as f:
                json.dump([int(x) for x in gt_bbox], f)

            index += 1


def train_epoch(model, save_dir: Path, data_dirs: list[Path], tb_writer, global_step, save):
    """
    tb_writer: TensorBoard SummaryWriter.
    global_step: Current global step for logging.
    """
    dataset = SimulatedDataset(data_dirs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.to(DEVICE)
    model.train()

    step = 0
    pbar = trange(STEPS_PER_EPOCH, desc="Training")
    while step < STEPS_PER_EPOCH:
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            tb_writer.add_scalar("train/loss", loss.item(), global_step + step)
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
            step += 1
            if step >= STEPS_PER_EPOCH:
                break
    pbar.close()

    if save:
        model_path = save_dir / "model.pt"
        print("Saving to", model_path)
        torch.save(model.state_dict(), model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Path to results directory.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data directory.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--save_every", type=int, default=5, help="Save every N epochs.")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=args.results / "logs")
    global_step = 0

    print("Begin training.")
    print("  Device:", DEVICE)
    print("  Results path:", args.results)

    videos_dataset = VideosDataset(args.data)
    print("Using videos from:", args.data)

    model = DiscamModel(MODEL_INPUT_RES).to(DEVICE)
    agent = Agent(model, VIDEO_RES, AGENT_VELOCITY)
    print("Model:")
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    prev_data_dirs = []
    for epoch in range(args.epochs):
        print("Begin epoch", epoch)
        epoch_path = args.results / f"epoch{epoch}"
        epoch_path.mkdir(parents=True, exist_ok=True)
        print("  Epoch path:", epoch_path)

        print("  Simulating...")
        data_dir = epoch_path / "data"
        simulate(videos_dataset, agent, epoch_path / "data")
        prev_data_dirs.append(data_dir)

        data_dirs = prev_data_dirs[max(0, epoch - DATA_HISTORY + 1):]
        print("  Training...")
        print("    Using data from:", data_dirs)
        train_epoch(model, epoch_path, data_dirs, tb_writer, global_step, (epoch + 1) % args.save_every == 0)
        global_step += STEPS_PER_EPOCH

        print("  End epoch", epoch)

        save_path = args.results / f"latest.pt"
        print("  Saving latest to", save_path)
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
