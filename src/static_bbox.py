"""
Calculate the bounding box, given the CV outputs.
"""

import cv2
import numpy as np

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

    def __init__(self):
        pass

    def dynamic_thres(self, img, thres):
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        return (img > thres).float()

    def update(self, detector_out):
        """
        Call once per frame, as this will track historical data.
        return {
            bbox: tuple of floats: x1, y1, x2, y2
        }
        """
        # Find min and max coords.
        xs = []
        ys = []
        for box in detector_out["filtered_bboxes"]:
            xs.append(box[0])
            xs.append(box[2])
            ys.append(box[1])
            ys.append(box[3])
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
