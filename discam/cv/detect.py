"""
Person detection and tracking module.
"""

import os

import numpy as np
import torch

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from ..utils.constants import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = os.path.dirname(os.path.abspath(__file__))


class Detector:
    """
    Detection with YOLO and SAHI. Tracking with ByteTrack.
    """

    def __init__(self):
        self.YOLO = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=os.path.join(ROOT, "yolo26n.pt"),
            confidence_threshold=0.2,
            device=DEVICE,
        )

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
            self.YOLO,
            slice_height=500,
            slice_width=500,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            verbose=0,
        )

        # Convert to (x, y, x, y)
        boxes = []
        for r in results.object_prediction_list:
            if r.category.id == 0 and r.score.value > 0.2:
                boxes.append(r.bbox.to_xyxy())
        boxes = np.array(boxes, dtype=int)
        return boxes
