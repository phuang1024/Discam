"""
Use detection bounding boxes and tracks
to determine how to move the camera view.

Run this file to test the tracking algorithm on a video.
"""

import argparse

import cv2

from detection import YoloTracker


class TrackAlgorithm:
    def __init__(self):
        self.tracker = YoloTracker()

    def step(self, frame):
        """
        return: (pan, tilt, zoom) control.
            Each is a value on the order of [-1, 1].
            Positive pan means pan right.
            Positive tilt means tilt down.
            Positive zoom means zoom in.
        """


class SimulatedCamera:
    """
    Simulated PTZ camera.
    """


def vis_algorithm():
    """
    Visualize tracking algorithm.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input video path.")
    args = parser.parse_args()

    tracker = YoloTracker()

    video = cv2.VideoCapture(args.input)
    while True:
        for _ in range(args.frame_skip):
            ret, frame = video.read()
        if not ret:
            break

        #frame = frame[250:750, 500:1500]

        result = tracker.step(frame)

        vis = draw_tracking(frame, tracker, result)
        cv2.imshow("track", vis)
        if cv2.waitKey(100) == ord("q"):
            break


if __name__ == "__main__":
    vis_algorithm()
