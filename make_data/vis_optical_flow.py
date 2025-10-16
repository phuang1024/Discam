import pickle

import cv2
import numpy as np

from utils import read_args



def vis_optical_flow(video_path, points):
    """
    Visualize salient points per frames.
    """
    video = cv2.VideoCapture(str(video_path))

    i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if i >= len(points):
            break

        for p in points[i]:
            cv2.circle(frame, tuple(p.astype(int)), 2, (0, 0, 255), -1)
        cv2.imshow("frame", frame)
        cv2.waitKey(10)

        i += 1


def main():
    args, bounds = read_args()
    with open(args.output / "points.pkl", "rb") as f:
        points = pickle.load(f)

    vis_optical_flow(args.video, points)


if __name__ == "__main__":
    main()
