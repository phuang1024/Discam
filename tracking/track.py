"""
YOLO detection and person tracking.
"""

from collections import deque

from ultralytics import YOLO

from .constants import *


class YoloTracker:
    """
    Incremental detection and tracking with YOLO.
    Keeps a queue of recent bbox centers for each unique person.
    Updates per frame.

    Detection and implicit tracking is performed on every call of step().
    On every Nth call, the results are appended to the result queues.
    Set this with track_interval.

    self.tracks has the results of tracking as discrete time series.
    Each element is (x, y, frame).
    Values are pixel coordinates of bounding box centers.
    """

    def __init__(self, track_interval):
        self.track_interval = track_interval

        self.model = YOLO("yolo26n.pt")
        self.tracks = {}
        self.iter = 0

    def step(self, frame, remove_lost=False):
        """
        Track people in frame and update in self.tracks.
        Returns results from detection.

        remove_lost: Whether to remove tracks that are no longer detected.
        """
        result = self.model.track(frame, imgsz=FRAME_RES, persist=True, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu()
        class_ids = result.boxes.cls.int().cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Check tracking interval.
        if self.iter % self.track_interval == 0:
            for box, cls, id in zip(boxes, class_ids, track_ids):
                # Check if is person.
                if cls != 0:
                    continue

                if id not in self.tracks:
                    self.tracks[id] = deque()

                # Append box center to track.
                x = ((box[0] + box[2]) / 2).item()
                y = ((box[1] + box[3]) / 2).item()
                self.tracks[id].append((x, y, self.iter))

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


def prepare_model_input(track, res):
    """
    Prepare track data for model by
    normalizing coordinates,
    computing velocity,
    padding,
    and converting to tensor.

    track: Discrete time series of (x, y) positions.
        E.g. from tracker.tracks[]
    res: Frame resolution.
    return: Tensor shape (TRACK_LEN, 5) with columns [mask, x, y, vx, vy].
    """
    # Normalize position.
    pos = torch.zeros([TRACK_LEN, 2], dtype=torch.float32)
    for i in range(len(track)):
        pos[i, 0] = track[i][0] / res[0]
        pos[i, 1] = track[i][1] / res[1]

    # Compute velocity as diff of consecutive.
    vel = torch.zeros([TRACK_LEN, 2], dtype=torch.float32)
    for i in range(len(track) - 1):
        vel[i] = pos[i + 1] - pos[i]

    # Make mask.
    mask = torch.zeros([TRACK_LEN, 1], dtype=torch.float32)
    mask[:len(track)] = 1

    data = torch.cat([mask, pos, vel], dim=1)
    return data
