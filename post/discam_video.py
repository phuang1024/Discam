"""
Run Discam on a video.
"""

import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(
    os.path.dirname(ROOT),
    "train",
))

import argparse
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from agent import Agent
from constants import AGENT_VELOCITY, DEVICE, MODEL_INPUT_RES
from model import DiscamModel


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    in_video = cv2.VideoCapture(str(args.input))
    width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = DiscamModel().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    agent = Agent(model, (width, height), velocity=AGENT_VELOCITY)
    agent.bbox = (0, 0, width, height)

    out_video = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        in_video.get(cv2.CAP_PROP_FPS),
        (width, height),
    )

    pbar = tqdm(total=int(in_video.get(cv2.CAP_PROP_FRAME_COUNT)))
    i = 0
    while True:
        ret, frame = in_video.read()
        if not ret:
            break
        pbar.update(1)
        i += 1

        if i > 2000:
            break

        # Extract current view
        bbox = tuple(map(int, agent.bbox))
        view = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        view = cv2.resize(view, (width, height))
        out_video.write(view)

        # Step agent.
        model_input = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        model_input = cv2.resize(model_input, MODEL_INPUT_RES)
        model_input = torch.from_numpy(model_input).permute(2, 0, 1).float() / 255.0
        model_input = model_input.to(DEVICE)
        agent.step(model_input)

    in_video.release()
    out_video.release()
    pbar.close()


if __name__ == "__main__":
    main()
