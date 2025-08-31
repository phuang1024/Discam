"""
Test a model.
"""

import argparse
import json
from pathlib import Path

import cv2
import torch

from agent import Agent
from constants import *
from model import DiscamModel


def draw_visualization(frame, gt, pred, crop: bool):
    """
    Draw an image to visualize the model.

    frame: Full image.
        ndarray, uint8, BGR
    gt: Ground truth bbox (x1, y1, x2, y2)
    pred: Predicted bbox

    If crop:
        Frame is cropped to model's prediction.

    Else:
        Original frame is kept. 
    """
    """
    crop = img[
        int(agent.bbox[1]) : int(agent.bbox[3]),
        int(agent.bbox[0]) : int(agent.bbox[2]),
    ]
    """

    cv2.rectangle(frame, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (pred[0], pred[1]), (pred[2], pred[3]), (0, 0, 255), 2)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Path to results directory.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data directory.")
    parser.add_argument("--crop", action="store_true", help="Show cropped output.")
    args = parser.parse_args()

    model = DiscamModel(MODEL_INPUT_RES)
    model.load_state_dict(torch.load(args.results, map_location=DEVICE))
    agent = Agent(model, VIDEO_RES, AGENT_VELOCITY)

    i = 0
    while True:
        img_path = args.data / f"{i}.frame.jpg"
        if not img_path.exists():
            break

        # Read image.
        img = cv2.imread(str(img_path))

        # Run agent step.
        img_t = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = cv2.resize(img_t, MODEL_INPUT_RES)
        img_t = torch.from_numpy(img_t).permute(2, 0, 1).float() / 255.0
        agent.step(img_t)

        # Read GT bbox.
        gt_path = args.data / f"{i}.allbbox.json"
        with open(gt_path, "r") as f:
            gt = json.load(f)

        vis = draw_visualization(img, gt, agent.bbox, args.crop)
        cv2.imshow("img", vis)
        cv2.waitKey(100)

        i += 1


if __name__ == "__main__":
    main()
