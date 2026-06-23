"""
Person detecting using RT-DETR.
Motion analysis using Farneback optical flow.
"""

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from bounding_box import extract_box, resize_bbox
from field_mask import read_mask, create_mask, create_persp_scale
from utils import *

YOLO_MODEL = YOLO("yolo26s.pt")


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
        _, _, _, players_fine, crop_box = self.run_yolo_twopass(frame)
        vis_detector(frame, players_fine, crop_box)
        return players_fine

    def run_yolo_twopass(self, frame):
        """
        Two pass detection. See Detector docs.
        """
        # First pass. Low person thres, high field mask thres.
        boxes_coarse = run_yolo_single(frame, 0.1)
        players_coarse = self.filter_boxes(boxes_coarse, 0.7)

        # Find frame overall bbox.
        box = extract_box(players_coarse, 150)
        if box is None:
            box = np.array((0, 0, frame.shape[1], frame.shape[0]), dtype=int)
        else:
            box = resize_bbox(box).astype(int)

        # Crop and second pass. High person thres, low field mask thres.
        frame_crop = frame[box[1]:box[3], box[0]:box[2]]
        boxes_fine = run_yolo_single(frame_crop, 0.1)
        # Correct coords.
        if len(boxes_fine) > 0:
            boxes_fine[:, [0, 2]] += box[0]
            boxes_fine[:, [1, 3]] += box[1]
        players_fine = self.filter_boxes(boxes_fine, 0.5)

        return boxes_coarse, players_coarse, boxes_fine, players_fine, box

    def filter_boxes(self, boxes, thres):
        """
        Filter boxes by field mask.
        boxes, return: boxes format.
        """
        ret = []
        for box in boxes:
            x1, y1, x2, y2 = box
            mid_x = (x1 + x2) // 2
            if self.blurred_mask[y2, mid_x] * self.persp_scale[y2, mid_x] > thres:
                ret.append(box)

        ret = np.array(ret, dtype=int)
        return ret


def run_yolo_single(frame, thres):
    """
    Run on single frame. Return person boxes.
    frame: cv2 format.
    return: boxes format.
    """
    results = YOLO_MODEL.predict(frame, conf=thres)
    boxes = results[0].boxes.xyxy.numpy().astype(int)
    return boxes


def vis_detector(frame, player_boxes, crop_box):
    frame = frame.copy()

    # Draw bboxes.
    """
    for x1, y1, x2, y2 in detector_out["boxes"].astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    """
    for x1, y1, x2, y2 in player_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw crop box.
    cv2.rectangle(frame, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (255, 0, 0), 2)

    # Overlay field mask
    """
    mask = detector_out["blurred_mask"] / 2 + 0.5
    mask = (mask * 255).astype(np.uint8)
    frame = cv2.addWeighted(frame, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    """

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)
