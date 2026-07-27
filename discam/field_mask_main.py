"""Manually drawn polygonal static field mask.

Mask format: ``ndarray float (N, 2)``, xy in ``[0, 1]`` relative to width and height.

Entry point to interactively create a mask.
"""

import argparse
import time

import cv2
import numpy as np

# Globals used in interactive mode.
_interactive_frame = None
# In coordinates of [0, 1] relative to W, H
_interactive_mask = []
_last_click = 0


def click_handler(event, x, y, flags, param):
    """Mouse click handler.
    Appends clicked point normalized to [0, 1].
    """
    global _interactive_frame, _interactive_mask, _last_click
    if event == cv2.EVENT_LBUTTONDOWN and time.time() - _last_click > 0.5:
        cv2.circle(_interactive_frame, (x, y), 3, (0, 0, 255), -1)
        x /= _interactive_frame.shape[1]
        y /= _interactive_frame.shape[0]
        _interactive_mask.append((x, y))

        _last_click = time.time()
    print(_interactive_mask)


def main():
    global _interactive_frame, _interactive_mask, _last_click

    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--frame", type=int, default=30, help="Frame number to use from video.")
    args = parser.parse_args()

    # Read frame.
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, _interactive_frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_handler)

    # Record click points.
    _last_click = time.time()
    while True:
        cv2.imshow("Frame", _interactive_frame)
        key = cv2.waitKey(100)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

    print("Writing to", args.output)
    mask = np.array(_interactive_mask)
    np.save(args.output, mask)
