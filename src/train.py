"""
Main training script.
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from tqdm import trange

from agent import Agent
from constants import *
from dataset import VideosDataset
from model import DiscamModel


@torch.no_grad()
def simulate(videos_dataset, agent, epoch_path: Path):
    """
    Simulate data for one epoch of training.
    """
    index = 0
    for sim in trange(SIMS_PER_EPOCH, desc="Simulating"):
        # Reset environment.
        frames, bboxes = videos_dataset.get_rand_chunk(
            SIM_STEPS * SIM_FRAME_SKIP,
            SIM_FRAME_SKIP,
            MODEL_INPUT_RES,
        )
        agent.bbox = bboxes[0].tolist()

        for step in range(SIM_STEPS):
            frame = frames[step]
            gt_bbox = bboxes[step]

            # Step agent.
            agent.step(frame)

            # Save frame and bbox.
            frame = frame.permute(1, 2, 0).numpy() * 255
            frame = frame.astype("uint8")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(epoch_path / f"{index}.frame.jpg"), frame)

            with open(epoch_path / f"{index}.pred.json", "w") as f:
                json.dump([int(x) for x in agent.bbox], f)

            with open(epoch_path / f"{index}.gt.pt", "w") as f:
                json.dump([int(x) for x in gt_bbox], f)

            index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Path to results directory.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data directory.")
    args = parser.parse_args()

    results_path = args.results
    sess_path = results_path / "test"

    videos_dataset = VideosDataset(args.data)

    model = DiscamModel(MODEL_INPUT_RES)
    agent = Agent(model, VIDEO_RES, AGENT_VELOCITY)

    epoch_path = sess_path / "0"
    epoch_path.mkdir(parents=True, exist_ok=True)

    simulate(videos_dataset, agent, epoch_path)


if __name__ == "__main__":
    main()
