"""
Calculate the bounding box, given the CV outputs.
"""

import torch

from utils import *


class StaticBBox:
    """
    The goal is:
    Given the current frame (i.e. pipeline output)
    and any historical information,
    calculate the ideal bounding box for this frame.

    This *does not* consider bounding box smoothness over time.
    This *does* consider results from past frames.

    DINO: Static thresholding.
    OF, BGR: Dynamic threshold:
        Images are normalized to 0-1 every frame, and compared to a fixed thres.
        Therefore, when there are a small amount of clear tracks, those will be highlighted.
        When not many strong tracks, much of the space will be highlighted.
    """

    def __init__(self):
        pass

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
        sim_mask = detector_out["mask"]

        # For now, find min and max coords of mask.
        ys, xs = torch.where(sim_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            x1 = x2 = y1 = y2 = 0
        else:
            x1 = xs.min().item() * 14
            x2 = xs.max().item() * 14
            y1 = ys.min().item() * 14
            y2 = ys.max().item() * 14

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
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("StaticBBox", frame)
    cv2.waitKey(1)
