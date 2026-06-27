"""
Person detection and tracking module.
"""

import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from utils.constants import *

YOLO = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.2,
)


class Detector:
    """
    Detection with YOLO and SAHI. Tracking with TODO.
    """

    def __init__(self):
        pass

    def update(self, frame):
        """
        frame: cv2 format.
        return: boxes format.
            Boxes of all detected people.
        """
        results = get_sliced_prediction(
            frame,
            YOLO,
            slice_height=500,
            slice_width=500,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            #verbose=0,
        )
        # Convert to boxes format.
        boxes = []
        for r in results.object_prediction_list:
            if r.category.id == 0 and r.score > 0.2:
                boxes.append(r.bbox.to_xyxy())
        boxes = np.array(boxes, dtype=int)
        return boxes
