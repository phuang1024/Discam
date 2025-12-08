"""
Analytic bounding box generator for data creation.

Run this script to generate bboxes for all frames of a video.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Parameters for computing frame difference.
FD_SCALE = 0.25
FD_ERODE_ITERS = 1
FD_THRES = 0.1


def vis_bbox(frame, bbox, color=(0, 255, 0)):
    """
    Draw bbox on frame and show.
    """
    frame = frame.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.imshow("bbox", frame)
    cv2.waitKey(10)


def compute_bboxes(video_path, step):
    """
    Compute bboxes on every nth frame independently.

    return: dict of frame index to bbox.
    """
    video = cv2.VideoCapture(str(video_path))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    bboxes = {}

    frame_idx = 0
    # This is updated in intervals of `step`.
    last_frame = None
    last_bbox = None

    pbar = tqdm(total=duration, desc="Computing bboxes")
    while True:
        for _ in range(step):
            ret, frame = video.read()
            frame_idx += 1
            pbar.update(1)
        if not ret:
            break

        orig_frame = frame.copy()
        frame = cv2.resize(frame, (int(width * FD_SCALE), int(height * FD_SCALE)))
        frame = frame.astype(np.float32) / 255.0

        if last_frame is not None:
            diff = np.abs(frame - last_frame)
            diff = np.mean(diff, axis=2)
            diff = (diff > FD_THRES).astype(np.uint8)
            diff = cv2.erode(diff, None, iterations=FD_ERODE_ITERS)

            ys, xs = np.where(diff > 0)
            if len(xs) > 0 and len(ys) > 0:
                x1 = int(np.min(xs) / FD_SCALE)
                x2 = int(np.max(xs) / FD_SCALE)
                y1 = int(np.min(ys) / FD_SCALE)
                y2 = int(np.max(ys) / FD_SCALE)
                bbox = (x1, x2, y1, y2)

            else:
                bbox = last_bbox

            if bbox is not None:
                bboxes[frame_idx - 1] = bbox
                vis_bbox(orig_frame, bbox)

            last_bbox = bbox

        last_frame = frame

    video.release()

    return bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to video file.")
    parser.add_argument("output", type=Path, help="Path to output directory.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    bboxes = compute_bboxes(args.input, step=5)


if __name__ == "__main__":
    main()
