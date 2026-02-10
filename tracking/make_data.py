"""
Generate and label data.

This is a three step process:
1. Run tracking on video footage.
2. Ask user to manually label trajectories.
3. Distill data for training.

The manually labeled data is more complicated than necessary for training:
- Tracks can and will be longer than the NN input length.
- There are more than 2 classes.
- Frame step is 1.
This is to be able to reuse the same set of labeled data on different training
configs, e.g. by lumping classes or truncating tracks.

Classes:
0: Active player. During a point.
1: Spectator. During a point.
2: Everyone. Between points.
3: Erroneous or unrelated track (e.g. person on different field).
"""

import sys
sys.path.append("..")

import argparse
import json
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from tracking import YoloTracker, prepare_model_input


def track(args):
    args.data.mkdir(exist_ok=True, parents=True)

    video = cv2.VideoCapture(args.video)
    tracker = YoloTracker(1)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Track for all frames.
    pbar = tqdm(total=length, desc="Tracking")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        tracker.step(frame, remove_lost=False)
        pbar.update(1)
    pbar.close()

    # Write tracks to file.
    print(f"{len(tracker.tracks)} tracks found.")
    for i, (id, track) in enumerate(tracker.tracks.items()):
        with open(args.data / f"{i}.meta.json", "w") as f:
            json.dump({
                "id": id,
                "frame_start": track[0][2],
            }, f)

        data = prepare_model_input(track, (width, height))
        torch.save(data, args.data / f"{i}.track.pt")


def label(args):
    pass


def distill(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to data directory.")
    subp = parser.add_subparsers(dest="command")

    track_p = subp.add_parser("track", help="Run tracking on video footage.")
    track_p.add_argument("--video", type=Path, required=True, help="Path to video file.")

    label_p = subp.add_parser("label", help="Manually label trajectories.")
    label_p.add_argument("--video", type=Path, required=True, help="Path to video file.")

    distill_p = subp.add_parser("distill", help="Distill data for training.")
    distill_p.add_argument("--output", type=Path, required=True, help="Path to output dir.")

    args = parser.parse_args()

    if args.command == "track":
        track(args)
    elif args.command == "label":
        label(args)
    elif args.command == "distill":
        distill(args)
