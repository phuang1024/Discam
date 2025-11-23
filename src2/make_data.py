"""
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def compute_optical_flow(args):
    """
    Computes an image (2D array) of pixel velocity vectors per frame.
    Sparse optical flow (LK) is computed every `of_frame_skip` frames, then interpolated.
    Each velocity vector image is an image of original resolution downsampled by `of_downsample` times.
        Shape (H, W, 2). dtype float.
    A pixel in the velocity image contains the velocity of that pixel in the corresponding frame.
    """
    video = cv2.VideoCapture(args.input)

    last_frame = None
    while True:
        for _ in range(args.of_frame_skip):
            ret, frame = video.read()
        if not ret:
            break

        if last_frame is not None:

        last_frame = frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to input video.")
    parser.add_argument("output", type=Path, help="Path to output dir.")
    parser.add_argument("--of_frame_skip", type=int, default=4,
        help="Frame skip when computing optical flow.")
    parser.add_argument("--of_downsample", type=float, default=2,
        help="Resolution downsampling of optical flow output.")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
