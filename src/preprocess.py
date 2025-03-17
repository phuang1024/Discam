import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


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

    cap = cv2.VideoCapture(str(file))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    last_frame = None
    mask = None
    for i, frame in tqdm(enumerate(generate_frames(args, cap)), desc=file.stem):
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
            min_x, min_y, max_x, max_y = x.min(), y.min(), x.max(), y.max()

            cv2.imwrite(str(args.output / file.stem / f"{i}.jpg"), frame)
            with open(args.output / file.stem / f"{i}.txt", "w") as f:
                f.write(f"{min_x} {min_y} {max_x} {max_y}")

        last_frame = frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--res", type=int, default=480, help="Output vertical resolution.")
    parser.add_argument("--fps", type=float, default=1, help="FPS of sampling.")
    args = parser.parse_args()

    for file in args.input.iterdir():
        if file.is_file() and file.suffix == ".mp4" and file.with_suffix(".txt").exists():
            process_file(args, file)


if __name__ == "__main__":
    main()
