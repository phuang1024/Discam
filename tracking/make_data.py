"""
Generate and label data.

This is a three step process:
1. Run tracking on video footage.
2. Ask user to manually label trajectories.
3. Distill data for training.

The manually labeled data is more complicated than necessary for training:
Tracks can and will be longer than the NN input length, and
there are more than 2 classes.
This is to be able to reuse the same set of labeled data on different training
configs, e.g. by lumping classes or truncating tracks.

Classes:
0: Active player. During a point.
1: Spectator. During a point.
2: Everyone. Between points.
3: Erroneous or unrelated track (e.g. person on different field).
"""

import argparse
from pathlib import Path


def track(args):
    pass


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
    track_p.add_argument("--video", type=Path, required=True, help="Path to video file.")

    distill_p = subp.add_parser("distill", help="Distill data for training.")
    distill_p.add_argument("--output", type=Path, required=True, help="Path to output dir.")

    args = parser.parse_args()

    if args.command == "track":
        track(args)
    elif args.command == "label":
        label(args)
    elif args.command == "distill":
        distill(args)
