"""
YOLO detection and person tracking.

Run this file to visualize bboxes and tracks.
"""

from collections import deque

from ultralytics import YOLO

from .constants import *


class YoloTracker:
    """
    Incremental detection and tracking with YOLO.
    Keeps a queue of recent bbox centers for each unique person.
    Updates per frame.

    This class respects TRACK_INTERVAL.
    A detection (and implicit tracking) is performed on every call of step().
    Only some iterations append to self.tracks.
    Therefore, to implement DETECT_INTERVAL, you should call step() every Nth frame.
    """

    def __init__(self):
        self.model = YOLO("yolo26n.pt")
        self.tracks = {}

        self.iter = 0

    def step(self, frame, remove_lost=False):
        """
        Track people in frame and update in self.tracks.
        Returns results from detection.

        remove_lost: Whether to remove tracks that are no longer detected.
        """
        result = self.model.track(frame, imgsz=FRAME_RES, persist=True)[0]
        boxes = result.boxes.xyxy.cpu()
        class_ids = result.boxes.cls.int().cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Check tracking interval.
        if self.iter % TRACK_INTERVAL == 0:
            for box, cls, id in zip(boxes, class_ids, track_ids):
                # Check if is person.
                if cls != 0:
                    continue

                if id not in self.tracks:
                    self.tracks[id] = deque()

                # Append box center to track.
                x = (box[0] + box[2]) / 2
                y = (box[1] + box[3]) / 2
                self.tracks[id].append((x, y))

                if len(self.tracks[id]) > TRACK_LEN:
                    self.tracks[id].popleft()

        # Remove lost tracks.
        if remove_lost:
            remove = []
            for key in self.tracks:
                if key not in track_ids:
                    remove.append(key)
            for key in remove:
                self.tracks.pop(key)

        self.iter += 1
        return result

    def clear_tracks(self):
        self.tracks = {}
