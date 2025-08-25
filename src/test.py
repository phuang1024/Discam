"""
Test a model.
"""

import argparse
from pathlib import Path

import cv2

from agent import Agent
from constants import *
from model import DiscamModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Path to results directory.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data directory.")
    args = parser.parse_args()

    model = DiscamModel(MODEL_INPUT_RES)
    agent = Agent(model, VIDEO_RES, MODEL_INPUT_RES, AGENT_VELOCITY)

    for i in range(1000):
        img = cv2.imread(str(args.data / f"{i}.frame.jpg"))
        agent.step(img)

        crop = img[
            int(agent.bbox[1]) : int(agent.bbox[3]),
            int(agent.bbox[0]) : int(agent.bbox[2]),
        ]

        cv2.imshow("crop", crop)
        cv2.waitKey(100)


if __name__ == "__main__":
    main()
