"""
YOLO person detection and tracking.

Run this file to visualize bboxes and tracks.
"""

import argparse
from hashlib import sha1

import cv2
from ultralytics import YOLO


class YoloTracker:
    max_track_len = 5

    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.tracks = {}

    def step(self, frame):
        """
        Track people in frame and update in self.tracks.
        Returns results from detection.
        """
        result = self.model.track(
            frame,
            imgsz=(1920, 1080),
            persist=True,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu()
        class_ids = result.boxes.cls.int().cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, cls, id in zip(boxes, class_ids, track_ids):
            # Check if is person.
            if cls != 0:
                continue

            if id not in self.tracks:
                self.tracks[id] = []

            # Append box center to track.
            x = (box[0] + box[2]) / 2
            y = (box[1] + box[3]) / 2
            self.tracks[id].append((x, y))

            if len(self.tracks[id]) > self.max_track_len:
                self.tracks[id].pop(0)

        """
        # Remove stale tracks.
        remove = []
        for key in self.tracks:
            if key not in track_ids:
                remove.append(key)
        for key in remove:
            self.tracks.pop(key)
        """

        return result

    def clear_tracks(self):
        """
        Clear all tracks.
        """
        self.tracks = {}


def get_color(id):
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

    boxes = result.boxes.xyxy.int().cpu()
    class_ids = result.boxes.cls.int().cpu().tolist()
    track_ids = result.boxes.id.int().cpu().tolist()
    for box, cls, id in zip(boxes, class_ids, track_ids):
        if cls == 0:
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                get_color(id),
                2,
            )

    for id, track in tracker.tracks.items():
        color = get_color(id)
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
    parser.add_argument("--frame_skip", type=int, default=5)
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
    vis_tracking()
