"""
Convert salient points obtained from optical flow
into bounding boxes.
Also writes frames as images.
"""

import json
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from utils import read_args


def compute_bbox(mask, points):
    min_y = mask.shape[0]
    max_y = 0
    min_x = mask.shape[1]
    max_x = 0
    for x, y in points:
        x = int(np.clip(x, 0, mask.shape[1] - 1))
        y = int(np.clip(y, 0, mask.shape[0] - 1))
        if mask[y, x]:
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
    return min_x, min_y, max_x, max_y


def write_bboxes(args, bounds, points):
    video = cv2.VideoCapture(str(args.video))

    mask = np.zeros((
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    ), dtype=np.uint8)
    pts = np.array(list(bounds.values()), dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    pbar = tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    i = 0
    bbox = None
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Write frame
        frame_path = args.output / f"{i}.jpg"
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

        # Write bbox
        if len(points[i]) > 0:
            bbox = compute_bbox(mask, points[i])
        assert bbox is not None, "No points found in the first frame"
        with open(args.output / f"{i}.json", "w") as f:
            json.dump(bbox, f)

        i += 1
        pbar.update(1)


def main():
    args, bounds = read_args()
    with open(args.output / "points.pkl", "rb") as f:
        points = pickle.load(f)

    write_bboxes(args, bounds, points)


if __name__ == "__main__":
    main()
