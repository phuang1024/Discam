"""
Make ground truth bounding box data of a video.
"""

import argparse
import json
from pathlib import Path

from optical_flow import chunked_optical_flow, vis_optical_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    bounds_path = args.video.with_suffix(".json")
    with open(bounds_path, "r") as f:
        bounds = json.load(f)

    points = chunked_optical_flow(args.video, bounds, 500)
    vis_optical_flow(args.video, points)


if __name__ == "__main__":
    main()
