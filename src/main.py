import argparse

import cv2

from constants import *
from pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    pipe = Pipeline()
    video = cv2.VideoCapture(args.video)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, RES)

        pipe.update(frame)
        stop


if __name__ == "__main__":
    main()
