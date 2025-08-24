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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Make ROI mask.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(list(roi.values()), dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mask_bool = mask.astype(bool)
    cv2.imwrite(str(args.output / "mask.png"), mask)

    # Process frames.
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
            if len(frame_queue) < args.compare_step:
                continue

            # Compute frame difference.
            frame_a = frame_queue[0]
            frame_b = frame_queue[-1]
            diff = cv2.absdiff(frame_a, frame_b)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_and(thresh, mask)
            thresh = cv2.medianBlur(thresh, 5)

            # Find XY min max.
            ys, xs = np.where(thresh > 0)
            if len(xs) == 0 or len(ys) == 0:
                # TODO
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Draw rectangle on frame.
            boxed_frame = frame.copy()
            cv2.rectangle(boxed_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("Motion", boxed_frame)
            cv2.waitKey(100)

            # Write frame
            write_index += 1

        if args.max_frames > 0 and write_index >= args.max_frames:
            break


if __name__ == "__main__":
    main()
