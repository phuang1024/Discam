"""
Calculate the bounding box, given the CV outputs.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from field_mask import read_mask, create_mask
from utils import *


class StaticBBox:
    """
    The goal is:
    Given the current frame (i.e. pipeline output)
    and any historical information,
    calculate the ideal bounding box for this frame.

    This *does not* consider bounding box smoothness over time.
    This *does* consider results from past frames.
    """

    def __init__(self, field_mask_path):
        self.field_mask_path = field_mask_path
        # ndarray [H, W] bool
        self.field_mask = create_mask(read_mask(self.field_mask_path))

    def dynamic_thres(self, img, thres):
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        return (img > thres).float()

    def update(self, detector_out, motion_out):
        """
        Call once per frame, as this will track historical data.
        return {
            bbox: tuple of floats: x1, y1, x2, y2
        }
        """
        # For each bbox in Detector's output, check if their feet in field mask.
        xs = []
        ys = []
        for box in detector_out["bboxes"]:
            bottom_x = int((box[0] + box[2]) / 2)
            bottom_y = int(box[3])
            bottom_x = np.clip(bottom_x, 0, self.field_mask.shape[1] - 1)
            bottom_y = np.clip(bottom_y, 0, self.field_mask.shape[0] - 1)
            if self.field_mask is None or self.field_mask[bottom_y, bottom_x] > 0:
                xs.append(box[0])
                xs.append(box[2])
                ys.append(box[1])
                ys.append(box[3])

        # Find min and max coords.
        if len(xs) == 0 or len(ys) == 0:
            x1 = x2 = y1 = y2 = 0
        else:
            x1 = min(xs)
            x2 = max(xs)
            y1 = min(ys)
            y2 = max(ys)

        return {
            "bbox": (x1, y1, x2, y2),
        }


def vis_static_bbox(frame, bbox_out):
    """
    frame: cv2 format
    bbox_out: Dict output of StaticBBox.update.
    """
    frame = frame.copy()

    # Draw box.
    box = bbox_out["bbox"]
    box = [int(x) for x in box]
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("StaticBBox", frame)
    cv2.waitKey(1)


def vis_field_mask(mask):
    """
    mask: ndarray [H, W] bool
    """
    vis = mask.astype(float) * 255
    cv2.imshow("Field Mask", vis)
    cv2.waitKey(0)
