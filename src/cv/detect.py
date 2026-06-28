"""
Person detection and tracking module.
"""

from argparse import Namespace

import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.engine.results import Boxes
from ultralytics.trackers import BYTETracker

from utils.constants import *

YOLO = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.2,
)


class Detector:
    """
    Detection with YOLO and SAHI. Tracking with ByteTrack.
    """

    def __init__(self):
        tracker_args = Namespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=5,
            match_thresh=0.8,
            fuse_score=True,
        )
        self.tracker = BYTETracker(tracker_args)

    def update(self, frame):
        """
        frame: cv2 format.
        return: ndarray int [N, 5], (x, y, x, y, track_id)
            Boxes of all detected people.
            If person is not tracked, track_id = -1
        """
        # Run SAHI.
        results = get_sliced_prediction(
            frame,
            YOLO,
            slice_height=500,
            slice_width=500,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            verbose=0,
        )

        # Convert to (x, y, x, y, conf, cls)
        boxes = []
        for r in results.object_prediction_list:
            if r.category.id == 0 and r.score.value > 0.2:
                box = r.bbox.to_xyxy()
                boxes.append((*box, r.score.value, 0))
        boxes = np.array(boxes, dtype=float)

        # Run tracker.
        boxes_ul = Boxes(boxes, (frame.shape[0], frame.shape[1]))
        tracks = self.tracker.update(boxes_ul)

        # Associate track and detection results.
        # Tracker will modify xyxy, so need to find closest match.
        ret = np.empty((boxes.shape[0], 5), dtype=float)
        if len(boxes) == 0:
            return ret.astype(int)

        ret[:, :4] = boxes[:, :4]
        ret[:, 4] = -1
        for track_box in tracks:
            best_score = 1e9
            # Index of best match in `boxes`.
            best_ind = -1
            for i, det_box in enumerate(boxes):
                score = np.linalg.norm(det_box[:4] - track_box[:4])
                if score < best_score:
                    best_score = score
                    best_ind = i

            assert best_ind != -1
            ret[best_ind, 4] = track_box[4]

        ret = ret.astype(int)
        return ret
