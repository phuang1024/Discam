"""
Script for testing and visualizing results.
"""

import sys
sys.path.append("..")

import argparse
from hashlib import sha1

import cv2

from tracking import *


def rand_color(id):
    """
    Get a random color based on id.
    RGB 0-255.
    """
    digest = int(sha1(str(id).encode()).hexdigest(), 16)
    return (
        digest % 256,
        (digest // 256) % 256,
        (digest // 65536) % 256,
    )


def draw_tracking(frame, tracker, result):
    """
    frame: Image.
    tracker: Tracker instance. Will access tracker.tracks.
    result: Detection results of frame.
    """
    frame = frame.copy()
    res = (frame.shape[1], frame.shape[0])

    # Draw boxes.
    boxes = result.boxes.xyxy.int().cpu()
    class_ids = result.boxes.cls.int().cpu().tolist()
    track_ids = result.boxes.id.int().cpu().tolist()
    for box, cls, id in zip(boxes, class_ids, track_ids):
        if cls == 0:
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                rand_color(id),
                2,
            )

    # Draw trajectories.
    for id, track in tracker.tracks.items():
        color = rand_color(id)
        for i in range(len(track) - 1):
            cv2.line(
                frame,
                (int(track[i][0]), int(track[i][1])),
                (int(track[i + 1][0]), int(track[i + 1][1])),
                color,
                3,
            )

    return frame


def vis_tracking():
    """
    Visualize YOLO tracking.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input video path.")
    parser.add_argument("--frame_skip", type=int, default=DETECT_INTERVAL)
    args = parser.parse_args()

    tracker = YoloTracker(TRACK_INTERVAL)

    video = cv2.VideoCapture(args.input)
    while True:
        for _ in range(args.frame_skip):
            ret, frame = video.read()
        if not ret:
            break

        result = tracker.step(frame, remove_lost=True)

        vis = draw_tracking(frame, tracker, result)
        cv2.imshow("track", vis)
        if cv2.waitKey(100) == ord("q"):
            break


if __name__ == "__main__":
    vis_tracking()
