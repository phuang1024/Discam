import argparse
import json
from pathlib import Path


def load_bounds(video_path):
    """
    Load bounds path.
    Same filename as video but with .json extension.
    """
    bounds_path = video_path.with_suffix(".json")
    with open(bounds_path, "r") as f:
        bounds = json.load(f)
    return bounds


def read_args():
    """
    Read command line arguments and load bounds.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    bounds = load_bounds(args.video)

    return args, bounds
