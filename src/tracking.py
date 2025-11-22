"""
YOLO detection and person tracking.
Frame motion estimation via feature matching.
"""

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
        result = self.model.track(frame, persist=True)[0]
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

        return result


def frame_motion(frame1, frame2):
    """
    Use dense feature matching to estimate the transform between two frames.
    """
