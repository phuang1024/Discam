import argparse
import time

import cv2

from constants import *
from pipeline import CVPipeline, vis_pipeline
from video_read import ScaledReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    pipe = CVPipeline()
    video = ScaledReader(args.video)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, RES)

        pipe.update(frame)
        vis_pipeline(pipe)


if __name__ == "__main__":
    main()
