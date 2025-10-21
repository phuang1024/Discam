"""
Generate bounding box data every N frames from a video.

Video must be a stationary camera.
Manually mark quadrilateral field region.
See README.md for more details.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from frame_diff import frame_diff, compute_bbox
from utils import EMA, bbox_aspect

Y_SALIENCE_SCALE = 3


def make_roi_mask(roi, width, height):
    """
    Render quadrilateral ROI.

    roi: Data read from JSON file.
    return: Mask image.
        ndarray uint8 (H, W) [0, 255]
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(list(roi.values()), dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def vis_bbox(frame, diff, bbox):
    """
    Visualize bounding box and differences.
    Creates cv2 window.

    frame: BGR image.
        ndarray uint8 (H, W, 3) [0, 255]
    diff: Frame difference.
        ndarray float32 (H, W) [0, 1]
    bbox: (x1, y1, x2, y2) bbox.
    """
    frame = frame.copy()

    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    color = (226, 99, 255)
    for ch in range(3):
        frame[..., ch] = color[ch] * diff + frame[..., ch] * (1 - diff)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


def process_frames(args, cap, bin_mask, scale_mask):
    """
    Process every N frames and save bounding box data.

    bin_mask: Binary mask of ROI.
    scale_mask: Linear scale mask from 1 to Y_SALIENCE_SCALE from bottom to top of field.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    read_index = 0
    write_index = 0
    frame_queue = deque(maxlen=args.compare_step)

    diff_ema = EMA()
    diff_mult = 2

    pbar = tqdm(total=total_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        read_index += 1
        frame_queue.append(frame)

        if read_index % args.frame_step == 0:
            frame1 = frame_queue[0]
            frame2 = frame_queue[-1]
            diff = frame_diff(frame1, frame2)
            diff = diff_ema.update(diff)
            diff = (diff * diff_mult).clip(0, 1)
            diff = diff * bin_mask

            bbox = compute_bbox(diff, scale_mask, thres=0.2)
            assert bbox is not None
            bbox = bbox_aspect(bbox, aspect=width / height, width=width, height=height)
            #vis_bbox(frame, diff, bbox)

            # Write frame
            with open(args.output / f"{read_index}.bbox.json", "w") as f:
                data = [int(v) for v in bbox]
                json.dump(data, f)
            write_index += 1

        if args.max_frames > 0 and write_index >= args.max_frames:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Path to input video file.")
    parser.add_argument("output", type=Path, help="Path to output directory.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of frames to process.")
    parser.add_argument("--frame_step", type=int, default=15, help="Process every Nth frame.")
    parser.add_argument("--compare_step", type=int, default=10, help="Frame step for motion comparison.")
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

    # Make ROI binary mask.
    bin_mask = make_roi_mask(
        roi,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    bin_mask = bin_mask.astype(bool).astype(np.float32)

    # Make linear scale mask
    scale_mask = np.empty(bin_mask.shape, dtype=np.float32)
    y_min = min(v[1] for v in roi.values())
    y_max = max(v[1] for v in roi.values())
    for y in range(scale_mask.shape[0]):
        value = np.interp(y, (y_min, y_max), (Y_SALIENCE_SCALE, 1))
        value = np.clip(value, 1, Y_SALIENCE_SCALE)
        scale_mask[y, :] = value

    process_frames(args, cap, bin_mask, scale_mask)


if __name__ == "__main__":
    main()
