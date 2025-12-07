"""
Analytic bounding box generator for data creation.

Run this script to generate bboxes for all frames of a video.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to video file.")
    parser.add_argument("output", type=Path, help="Path to output directory.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
