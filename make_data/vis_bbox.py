"""
Visualize bboxes.
"""

import json

import cv2
import numpy as np

from utils import read_args


def vis_bboxes(args):
    i = 0
    while True:
        frame_path = args.output / f"{i}.jpg"
        bbox_path = args.output / f"{i}.json"
        if not frame_path.exists() or not bbox_path.exists():
            break

        frame = cv2.imread(str(frame_path))
        with open(bbox_path, "r") as f:
            bbox = json.load(f)

        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2,
        )
        cv2.imshow("frame", frame)
        cv2.waitKey(10)

        i += 1


def main():
    args, bounds = read_args()
    vis_bboxes(args)


if __name__ == "__main__":
    main()
