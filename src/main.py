import argparse

import cv2
import torch

from pipeline import Pipeline
from video_read import ScaledReader

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    video = ScaledReader(args.video)
    pipe = Pipeline()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        pipe.update(frame)


if __name__ == "__main__":
    main()
