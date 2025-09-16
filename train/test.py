"""
Test and visualize model in a simulated environment.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from agent import Agent
from constants import *
from dataset import compute_edge_weights
from model import DiscamModel


def draw_edge_weights(frame, weights, bbox, color, size=50, offset=0):
    """
    Draw perpendicular lines on bbox edge to indicate edge weights.

    frame: BGR image.
        ndarray uint8 (H, W, 3) [0, 255]
    weights: Edge weights in order: (up, right, down, left)
    bbox: (x1, y1, x2, y2)
    color: BGR color tuple.
    size: Maximum length of line.
    offset: Offset parallel to edge (to draw multiple sets of weights without overlap).
    """
    def draw_line(p1, p2):
        cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 2)

    # Up
    p1 = (np.array((bbox[0], bbox[1])) + np.array((bbox[2], bbox[1]))) / 2
    p1[0] += offset
    p2 = p1 + np.array((0, -size)) * weights[0]
    draw_line(p1, p2)

    # Right
    p1 = (np.array((bbox[2], bbox[1])) + np.array((bbox[2], bbox[3]))) / 2
    p1[1] += offset
    p2 = p1 + np.array((size, 0)) * weights[1]
    draw_line(p1, p2)

    # Down
    p1 = (np.array((bbox[0], bbox[3])) + np.array((bbox[2], bbox[3]))) / 2
    p1[0] += offset
    p2 = p1 + np.array((0, size)) * weights[2]
    draw_line(p1, p2)

    # Left
    p1 = (np.array((bbox[0], bbox[1])) + np.array((bbox[0], bbox[3]))) / 2
    p1[1] += offset
    p2 = p1 + np.array((-size, 0)) * weights[3]
    draw_line(p1, p2)


def draw_visualization(frame, gt_bbox, agent_bbox, pred_edge_weights, crop: bool):
    """
    Draw an image to visualize the model.

    frame: Full BGR image.
        ndarray uint8 (H, W, 3) [0, 255]
    gt: Ground truth bbox (x1, y1, x2, y2)
    pred: Predicted bbox
    crop: See below.

    If crop:
        Frame is cropped to model's prediction.
    Else:
        Original frame is kept. 
    """
    if crop:
        frame = frame[
            int(agent_bbox[1]) : int(agent_bbox[3]),
            int(agent_bbox[0]) : int(agent_bbox[2]),
        ]

    else:
        cv2.rectangle(frame, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (agent_bbox[0], agent_bbox[1]), (agent_bbox[2], agent_bbox[3]), (0, 0, 255), 2)

        draw_edge_weights(frame, pred_edge_weights, agent_bbox, (0, 0, 255), offset=2)
        gt_edge_weights = compute_edge_weights(agent_bbox, gt_bbox).numpy()
        draw_edge_weights(frame, gt_edge_weights, agent_bbox, (0, 255, 0))

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data dir of a single video.")
    parser.add_argument("--crop", action="store_true", help="Whether to show cropped output.")
    parser.add_argument("--start_from_gt", action="store_true", help="Start simulation from GT bbox.")
    parser.add_argument("--start_from_random", action="store_true", help="Start simulation from random bbox.")
    args = parser.parse_args()

    model = DiscamModel(MODEL_INPUT_RES, EDGE_WEIGHT_TEMP)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
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
        agent_bbox = tuple(map(int, agent.bbox))
        img_t = img_t[agent_bbox[1] : agent_bbox[3], agent_bbox[0] : agent_bbox[2]]
        img_t = cv2.resize(img_t, MODEL_INPUT_RES)
        img_t = torch.from_numpy(img_t).permute(2, 0, 1).float() / 255.0
        pred_edge_weights = agent.step(img_t)

        # Read GT bbox.
        gt_path = args.data / f"{i}.allbbox.json"
        with open(gt_path, "r") as f:
            gt_bbox = json.load(f)

        if args.start_from_gt and i == 0:
            agent.set_bbox(gt_bbox)
        elif args.start_from_random and i == 0:
            x1 = np.random.randint(0, VIDEO_RES[0] - 100)
            y1 = np.random.randint(0, VIDEO_RES[1] - 100)
            x2 = np.random.randint(x1 + 50, VIDEO_RES[0])
            y2 = np.random.randint(y1 + 50, VIDEO_RES[1])
            agent.set_bbox((x1, y1, x2, y2))

        agent_bbox = tuple(map(int, agent.bbox))
        vis = draw_visualization(img, gt_bbox, agent_bbox, pred_edge_weights, args.crop)
        cv2.imshow("img", vis)
        cv2.waitKey(1)

        i += 1


if __name__ == "__main__":
    main()
