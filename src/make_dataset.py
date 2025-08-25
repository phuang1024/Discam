"""
Generate training data from a video.

Video must be a stationary camera.
Manually mark quadrilateral field region.
Detects and bounds motion in the ROI.

TODOS:
- Detect and account for camera shake. Don't count it as motion.
- Detect stationary people and include in bounding box.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from frame_diff import frame_diff


def make_roi_mask(roi, width, height):
    """
    Draw quadrilateral ROI.

    roi: Data read from JSON file.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(list(roi.values()), dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    #mask_bool = mask.astype(bool)
    return mask


def process_frames(args, cap, mask):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    read_index = 0
    write_index = 0
    frame_queue = deque(maxlen=args.compare_step)

    pbar = tqdm(total=total_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        read_index += 1
        frame_queue.append(frame)

        if read_index % args.frame_step == 0:
            # TODO
            frame1 = frame_queue[0]
            frame2 = frame_queue[-1]

            frame_diff(frame1, frame2)

            # Write frame
            write_index += 1

        if args.max_frames > 0 and write_index >= args.max_frames:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Path to input video file.")
    parser.add_argument("output", type=Path, help="Path to output directory.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of frames to process.")
    parser.add_argument("--frame_step", type=int, default=5, help="Process every Nth frame.")
    parser.add_argument("--compare_step", type=int, default=8, help="Frame step for motion comparison.")
    args = parser.parse_args()

    # Read ROI data.
    json_path = args.video.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        roi = json.load(f)

    # Create output directory.
    args.output.mkdir(parents=True, exist_ok=True)

    # Open video file.
    cap = cv2.VideoCapture(str(args.video))

    # Make ROI mask.
    mask = make_roi_mask(
        roi,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cv2.imwrite(str(args.output / "mask.png"), mask)

    process_frames(args, cap, mask)


if __name__ == "__main__":
    main()
