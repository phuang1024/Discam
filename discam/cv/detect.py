"""Detection module.
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
    """Person detection with YOLO and SAHI.
    """

    def __init__(self):
        self.YOLO = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=os.path.join(ROOT, "yolo26n.pt"),
            confidence_threshold=DET_THRES,
            device=DEVICE,
        )

    def update(self, frame):
        """
        Args:
            frame: ``cv2 format``.

        Returns:
            ``boxes format``.
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

        # Convert to xyxy.
        boxes = []
        for r in results.object_prediction_list:
            if r.category.id == 0 and r.score.value > DET_THRES:
                boxes.append(r.bbox.to_xyxy())
        boxes = np.array(boxes, dtype=int)
        return boxes
