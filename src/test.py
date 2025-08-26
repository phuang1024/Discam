"""
Test a model.
"""

import argparse
from pathlib import Path

import cv2
import torch

from agent import Agent
from constants import *
from model import DiscamModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Path to results directory.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data directory.")
    args = parser.parse_args()

    model = DiscamModel(MODEL_INPUT_RES)
    agent = Agent(model, VIDEO_RES, AGENT_VELOCITY)

    for i in range(1000):
        img = cv2.imread(str(args.data / f"{i}.frame.jpg"))

        img_t = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = cv2.resize(img_t, MODEL_INPUT_RES)
        img_t = torch.from_numpy(img_t).permute(2, 0, 1).float() / 255.0
        agent.step(img_t)

        crop = img[
            int(agent.bbox[1]) : int(agent.bbox[3]),
            int(agent.bbox[0]) : int(agent.bbox[2]),
        ]

        cv2.imshow("crop", crop)
        cv2.waitKey(100)


if __name__ == "__main__":
    main()
