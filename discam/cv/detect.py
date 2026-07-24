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

    def __init__(self, live_mode):
        self.live_mode = live_mode
        self.YOLO = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=os.path.join(ROOT, "yolo26n.pt"),
            confidence_threshold=DET_THRES,
            device=DEVICE,
        )

    def update(self, frame, ptz=None):
        """
        Args:
            frame: ``cv2 format``.
            ptz: ``(p, t, z)`` for live mode, to account for shifted mask.

        Returns:
            ``(boxes, adj_boxes)``, both ``boxes format``.
            - ``boxes``: Detected boxes in coords of given ``frame``.
            - ``adj_boxes``: Boxes in coords of original camera view (zero PTZ).
              Is ``None`` if ``ptz == None``.
        """
        # Run SAHI.
        tile_size = TILE_SIZE_LIVE if self.live_mode else TILE_SIZE_POST
        results = get_sliced_prediction(
            frame,
            self.YOLO,
            slice_height=tile_size,
            slice_width=tile_size,
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

        # Calculate boxes adjusted for PTZ.
        adj_boxes = None
        if ptz is not None:
            adj_boxes = self.apply_ptz(boxes, ptz)
        return boxes, adj_boxes

    def apply_ptz(self, boxes, ptz):
        """Given ``boxes`` in coords of current view,
        return boxes in coords of original (p=t=0, z=1) view.
        TODO current alg isn't perfect.

        Args:
            boxes: ``boxes format``.
            ptz: ``ptz format``.
        """
        boxes = boxes.copy()
        # Apply inverse zoom.
        mid_x = CV_RES[0] / 2
        for i in (0, 2):
            boxes[:, i] = mid_x + (boxes[:, i] - mid_x) / ptz[2]
        mid_y = CV_RES[1] / 2
        for i in (1, 3):
            boxes[:, i] = mid_y + (boxes[:, i] - mid_y) / ptz[2]

        # Apply inverse pan tilt.
        px_per_deg = CV_RES[0] / CAM_FOV
        for i in (0, 2):
            boxes[:, i] = boxes[:, i] + px_per_deg * ptz[0]
        for i in (1, 3):
            boxes[:, i] = boxes[:, i] + px_per_deg * ptz[1]
        return boxes
