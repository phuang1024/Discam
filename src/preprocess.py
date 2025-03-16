import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_frames(args, cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / args.fps)

    while True:
        for i in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                return
        frame = cv2.resize(frame, (int(frame.shape[1] * args.res / frame.shape[0]), args.res))
        yield frame


def process_file(args, file):
    (args.output / file.stem).mkdir(parents=True, exist_ok=True)

    with open(file.with_suffix(".txt"), "r") as f:
        coords = np.array(list(map(int, f.read().strip().split()))).reshape(-1, 2)
        print(coords)

    cap = cv2.VideoCapture(str(file))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    last_frame = None
    mask = None
    last_bounds = None
    for i, frame in enumerate(generate_frames(args, cap)):
        if i == 0:
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.fillPoly(mask, [coords], (255, 255, 255))
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask = (mask > 0).astype(np.uint8)
        else:
            diff = cv2.absdiff(frame, last_frame) * mask
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            salient = diff > 25
            y, x = np.where(salient)
            if len(x) > 100:
                min_x, min_y, max_x, max_y = x.min() - args.margin, y.min() - args.margin, x.max() + args.margin, y.max() + args.margin
                min_x = np.clip(min_x, 0, frame.shape[1] - 1)
                min_y = np.clip(min_y, 0, frame.shape[0] - 1)
                max_x = np.clip(max_x, 0, frame.shape[1] - 1)
                max_y = np.clip(max_y, 0, frame.shape[0] - 1)
                bounds = np.array([min_x, min_y, max_x, max_y])
                if last_bounds is not None:
                    bounds = (last_bounds * 0.9 + bounds * 0.1).astype(int)
                last_bounds = bounds

                cv2.imwrite(str(args.output / file.stem / f"{i}.jpg"), frame)
                with open(args.output / file.stem / f"{i}.txt", "w") as f:
                    f.write(" ".join(map(str, bounds)))

                frame_vis = frame.copy()
                cv2.rectangle(frame_vis, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 255, 0), 2)
                cv2.imshow("frame", frame_vis)
                cv2.waitKey(0)

        last_frame = frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--res", type=int, default=480, help="Output vertical resolution.")
    parser.add_argument("--fps", type=float, default=1, help="FPS of sampling.")
    parser.add_argument("--margin", type=int, default=30)
    args = parser.parse_args()

    for file in args.input.iterdir():
        if file.is_file() and file.suffix == ".mp4" and file.with_suffix(".txt").exists():
            print(f"Processing {file.name}")
            process_file(args, file)


if __name__ == "__main__":
    main()
