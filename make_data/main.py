"""
Make ground truth bounding box data of a video.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2

from optical_flow import chunked_optical_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    bounds_path = args.video.with_suffix(".json")
    with open(bounds_path, "r") as f:
        bounds = json.load(f)

    chunked_optical_flow(args.video, bounds)


if __name__ == "__main__":
    main()
