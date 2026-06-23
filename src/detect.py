"""
Person detecting using RT-DETR.
Motion analysis using Farneback optical flow.
"""

import cv2
import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from field_mask import read_mask, create_mask, create_persp_scale
from utils import *

YOLO = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.1,
)


class Detector:
    """
    Person detection with YOLO, and spectator classification.

    2 step detection:
    First detection.
    Crop frame around players for better res. Detect again with higher threshold.
    """

    def __init__(self, field_mask_path):
        mask_points = read_mask(field_mask_path)
        self.field_mask = create_mask(mask_points).astype(np.float32)
        # Is a measure of closeness to border. -1 outside, 1 inside, 0 on border.
        self.blurred_mask = cv2.blur(self.field_mask, (FIELD_MASK_BLUR, FIELD_MASK_BLUR))
        self.blurred_mask = 2 * self.blurred_mask - 1

        # Scale to account for far people being small. 1 near, 3 far.
        self.persp_scale = create_persp_scale(mask_points)

    def update(self, frame):
        """
        frame: cv2 format.
        return: boxes format.
        """
        results = get_sliced_prediction(
            frame,
            YOLO,
            slice_height=500,
            slice_width=500,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )
        boxes = []
        for r in results.object_prediction_list:
            if r.category.id == 0 and r.score > 0.2:
                x1, y1, x2, y2 = r.bbox.to_xyxy()
                mid_x = int((x1 + x2) / 2)
                if self.blurred_mask[int(y2), mid_x] * self.persp_scale[int(y2), mid_x] > 0.5:
                    boxes.append((x1, y1, x2, y2))

        boxes = np.array(boxes, dtype=int)
        vis_detector(frame, boxes)
        return boxes


def vis_detector(frame, player_boxes):
    frame = frame.copy()

    # Draw bboxes.
    """
    for x1, y1, x2, y2 in detector_out["boxes"].astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    """
    for x1, y1, x2, y2 in player_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay field mask
    """
    mask = detector_out["blurred_mask"] / 2 + 0.5
    mask = (mask * 255).astype(np.uint8)
    frame = cv2.addWeighted(frame, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    """

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)
