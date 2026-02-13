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


def get_color_nn(tracker, model, res):
    data = []
    ids = list(tracker.tracks.keys())
    for id in ids:
        data.append(prepare_model_input(tracker.tracks[id], res))

    data = torch.stack(data, dim=0).to(DEVICE)
    data = data.permute(0, 2, 1)
    cls = model(data).argmax(dim=1).cpu().tolist()

    colors = {}
    for id, c in zip(ids, cls):
        if c == 0:
            colors[id] = (0, 255, 0)
        else:
            colors[id] = (255, 255, 255)

    return colors


def draw_tracking(frame, tracker, result, model):
    """
    frame: Image.
    tracker: Tracker instance. Will access tracker.tracks.
    result: Detection results of frame.
    """
    frame = frame.copy()
    res = (frame.shape[1], frame.shape[0])

    # Get color for each ID. Based on model classification, or random if no model.
    if model is None:
        colors = {id: rand_color(id) for id in tracker.tracks.keys()}
    else:
        colors = get_color_nn(tracker, model, res)

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
                colors.get(id, (0, 0, 0)),
                2,
            )

    # Draw trajectories.
    for id, track in tracker.tracks.items():
        for i in range(len(track) - 1):
            cv2.line(
                frame,
                (int(track[i][0]), int(track[i][1])),
                (int(track[i + 1][0]), int(track[i + 1][1])),
                colors.get(id, (0, 0, 0)),
                3,
            )

    return frame


def vis_tracking():
    """
    Visualize YOLO tracking.
    Query NN and color track based on class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input video path.")
    parser.add_argument("--frame_skip", type=int, default=DETECT_INTERVAL)
    parser.add_argument("--model")
    args = parser.parse_args()

    model = None
    if args.model is not None:
        model = TrackClassifier().to(DEVICE)
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    tracker = YoloTracker(TRACK_INTERVAL, TRACK_LEN)

    video = cv2.VideoCapture(args.input)
    while True:
        for _ in range(args.frame_skip):
            ret, frame = video.read()
        if not ret:
            break

        result = tracker.step(frame, remove_lost=False)

        vis = draw_tracking(frame, tracker, result, model)
        cv2.imshow("track", vis)
        if cv2.waitKey(100) == ord("q"):
            break


if __name__ == "__main__":
    vis_tracking()
