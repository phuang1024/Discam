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
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tracking import *

# Minimum number of points in track to add to dataset.
MIN_TRACK_LEN = 10
# When distilling, number of samples per track per TRACK_LEN by class.
NUM_SAMPLES = {
    0: 1,
    1: 8,
}


def track(args):
    args.data.mkdir(exist_ok=True, parents=True)

    video = cv2.VideoCapture(args.video)
    tracker = YoloTracker(1, float("inf"))

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

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
    cumul_track_len = 0
    i = 0
    for id, track in tracker.tracks.items():
        if len(track) < MIN_TRACK_LEN:
            continue

        with open(args.data / f"{i}.meta.json", "w") as f:
            json.dump({
                "id": id,
                "frame_start": track[0][2],
                "frame_end": track[-1][2],
            }, f, indent=4)

        with open(args.data / f"{i}.track.json", "w") as f:
            json.dump(list(track), f, indent=4)

        cumul_track_len += len(track)
        i += 1

    print(f"{i} tracks found.")
    print(f"Average track length: {cumul_track_len / i:.2f} frames.")


def label(args):
    # Find indices to label.
    total = 0
    to_label = []
    for file in args.data.iterdir():
        if ".track.json" in file.name:
            total += 1
            name = file.stem.split(".")[0]
            if not (args.data / f"{name}.label.txt").exists():
                to_label.append(name)

    print(f"Found {total} tracks, {len(to_label)} to label.")

    video = cv2.VideoCapture(args.video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for name in to_label:
        print(f"Labeling {name}.")
        with open(args.data / f"{name}.meta.json") as f:
            meta = json.load(f)

        # Get frame at end.
        video.set(cv2.CAP_PROP_POS_FRAMES, meta["frame_end"])
        ret, frame = video.read()
        if not ret:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw track on frame.
        with open(args.data / f"{name}.track.json") as f:
            track = json.load(f)
        track = np.array(track).astype(int)
        cv2.circle(frame, (track[-1][0], track[-1][1]), 15, (0, 0, 255), -1)
        for i in range(len(track) - 1):
            cv2.line(frame, (track[i][0], track[i][1]), (track[i + 1][0], track[i + 1][1]), (0, 255, 0), 5)

        # Ask for label.
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("label", frame)
        key = cv2.waitKey(0)
        label = None
        for i in range(4):
            if key == ord(str(i)):
                label = i
                break
        else:
            print("Invalid key, skipping.")

        # Write label to file.
        if label is not None:
            print("Label:", label)
            with open(args.data / f"{name}.label.txt", "w") as f:
                f.write(str(label))
                f.write("\n")


def distill(args):
    video = cv2.VideoCapture(args.video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    ids = []
    for file in args.data.iterdir():
        if ".label.txt" in file.name:
            name = file.stem.split(".")[0]
            ids.append(name)
    print(f"Found {len(ids)} labeled tracks.")

    write_idx = 0
    for id in tqdm(ids):
        with open(args.data / f"{id}.label.txt") as f:
            label = int(f.read().strip())

        with open(args.data / f"{id}.track.json") as f:
            track = json.load(f)
        track = torch.tensor(track)[:, :2]

        # Sample multiple datas per track, based on class imbalance and track length.
        num_samples = NUM_SAMPLES.get(label, 1)
        indices = range(
            0,
            max(track.shape[0] - TRACK_LEN * TRACK_INTERVAL, 0),
            TRACK_INTERVAL,
        )
        # Uniformly spaced start index.
        for start_i in indices:
            # Supersampling index, step 1.
            for step_i in range(num_samples):
                curr_len = random.randint(TRACK_LEN // 2, TRACK_LEN)
                curr_track = track[start_i + step_i : start_i + step_i + curr_len * TRACK_INTERVAL : TRACK_INTERVAL]
                assert curr_track.shape[0] <= TRACK_LEN
                curr_track = prepare_model_input(curr_track, (width, height))

                # Write to file.
                torch.save(curr_track, args.output / f"{write_idx}.pt")
                with open(args.output / f"{write_idx}.label.txt", "w") as f:
                    f.write(str(label))
                    f.write("\n")

                write_idx += 1

    print(f"Wrote {write_idx} training samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to data directory.")
    parser.add_argument("--video", type=Path, required=True, help="Path to video file.")
    subp = parser.add_subparsers(dest="command")

    track_p = subp.add_parser("track", help="Run tracking on video footage.")
    label_p = subp.add_parser("label", help="Manually label trajectories.")
    distill_p = subp.add_parser("distill", help="Distill data for training.")
    distill_p.add_argument("--output", type=Path, required=True, help="Path to output dir.")

    args = parser.parse_args()

    if args.command == "track":
        track(args)
    elif args.command == "label":
        label(args)
    elif args.command == "distill":
        args.output.mkdir(exist_ok=True, parents=True)
        distill(args)
