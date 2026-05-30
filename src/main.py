import argparse

import torch

from pipeline import Pipeline
from video_read import ScaledReader

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--field_mask")
    args = parser.parse_args()

    video = ScaledReader(args.video)
    pipe = Pipeline(args.field_mask)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        pipe.update(frame)


if __name__ == "__main__":
    main()
