"""
Manual quadrilateral static field mask.
Mask is 4 points in 2D.

Run this file to interactively create a mask.
"""

import argparse
import time

import cv2
import numpy as np

from utils import *

_interactive_frame = None
# In coordinates of [0, 1] relative to W, H
_interactive_mask = []
_last_click = 0


def write_mask(mask, path):
    """Write mask to file."""
    np.save(path, mask)


def read_mask(path):
    """Read mask from file."""
    return np.load(path)


def create_mask(points, res=RES):
    """
    Draw mask binary image.
    points: From read_mask()
    res: (W, H) resolution.
    return: ndarray bool [H, W]
    """
    points[:, 0] *= res[0]
    points[:, 1] *= res[1]
    points = points.astype(int)

    mask = np.zeros(res[::-1], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    mask = mask.astype(bool)
    return mask


def create_persp_scale(points, res=RES, max_val=3):
    """
    Create per-pixel scaling factor to account for far people being small.
    Y axis lerp: From (min(points y coord) to yres - 1), to (max_val to 1).
    return: ndarray float [H, W], from 1 to max_val.
    """
    img = np.zeros(res[::-1], dtype=float)
    min_y = points[:, 1].min()
    for y in range(res[1]):
        if y < min_y:
            img[y] = max_val
        else:
            img[y] = np.interp(y, [min_y, res[1] - 1], [max_val, 1])

    return img


def click_handler(event, x, y, flags, param):
    """Mouse click handler."""
    global _interactive_frame, _interactive_mask, _last_click

    if event == cv2.EVENT_LBUTTONDOWN and time.time() - _last_click > 0.5:
        cv2.circle(_interactive_frame, (x, y), 5, (0, 0, 255), -1)
        x = x / _interactive_frame.shape[1]
        y = y / _interactive_frame.shape[0]
        _interactive_mask.append((x, y))

        _last_click = time.time()

    print(_interactive_mask)


def main():
    global _interactive_frame, _interactive_mask, _last_click

    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--frame", type=int, default=30)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, _interactive_frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return
    _interactive_frame = cv2.resize(_interactive_frame, None, fx=0.5, fy=0.5)

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_handler)

    _last_click = time.time()
    while True:
        cv2.imshow("Frame", _interactive_frame)
        cv2.waitKey(100)
        if len(_interactive_mask) == 4:
            break

    # Un-scale.
    mask = np.array(_interactive_mask)

    print("Writing to", args.output)
    write_mask(mask, args.output)


if __name__ == "__main__":
    main()
