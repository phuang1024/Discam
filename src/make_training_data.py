"""
Generate training data from bounding box data and video.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import EMA


def interploate_bboxes(bboxes, total_frames):
    """
    Interpolate bboxes between given bboxes.
    Indices range from 0 to total_frames - 1.
    """
    keys = sorted(bboxes.keys())

    ret = np.zeros((total_frames, 4), dtype=np.int32)

    for i in range(total_frames):
        if i <= keys[0]:
            ret[i] = bboxes[keys[0]]
        elif i >= keys[-1]:
            ret[i] = bboxes[keys[-1]]
        else:
            for j in range(len(keys) - 1):
                if keys[j] <= i <= keys[j + 1]:
                    ratio = (i - keys[j]) / (keys[j + 1] - keys[j])
                    ret[i] = (bboxes[keys[j]] * (1 - ratio) + bboxes[keys[j + 1]] * ratio).astype(np.int32)
                    break

    # Add an EMA.
    ema = EMA(alpha=0.96)
    for i in range(total_frames):
        ret[i] = ema.update(ret[i])

    return ret


def vis_crop(cap, bboxes):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        x1, y1, x2, y2 = bboxes[i]
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (960, 540))

        cv2.imshow("Crop", crop)
        cv2.waitKey(30)


def write_frames(args, cap, bboxes):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=total_frames)
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        cv2.imwrite(str(args.output / f"{i}.frame.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        with open(args.output / f"{i}.allbbox.json", "w") as f:
            data = [int(v) for v in bboxes[i]]
            json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Path to input video file.")
    parser.add_argument("output", type=Path, help="Path to output directory.")
    args = parser.parse_args()

    # Read bbox data
    bboxes = {}
    for file in args.output.glob("*.bbox.json"):
        key = int(file.stem.split(".")[0])
        with open(file, "r") as f:
            bbox = np.array(json.load(f), dtype=np.int32)
        bboxes[key] = bbox

    # Open video file.
    cap = cv2.VideoCapture(str(args.video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bboxes_interp = interploate_bboxes(bboxes, total_frames)

    #vis_crop(cap, bboxes_interp)
    write_frames(args, cap, bboxes_interp)


if __name__ == "__main__":
    main()
