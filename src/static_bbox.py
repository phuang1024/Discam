"""
Calculate the bounding box, given the CV outputs.
"""

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

    DINO: Static thresholding.
    OF, BGR: Dynamic threshold:
        Images are normalized to 0-1 every frame, and compared to a fixed thres.
        Therefore, when there are a small amount of clear tracks, those will be highlighted.
        When not many strong tracks, much of the space will be highlighted.
    """

    def __init__(self, field_mask_path):
        self.field_mask_path = field_mask_path
        # Tensor [H', W'] float 0-1
        self.field_mask = None

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

        if self.field_mask_path is not None and self.field_mask is None:
            # Load and resize mask on first iter.
            self.field_mask = create_mask(read_mask(self.field_mask_path))
            self.field_mask = torch.from_numpy(self.field_mask).float()
            self.field_mask = F.interpolate(self.field_mask[None, None, ...], sim_mask.shape)[0, 0]

        if self.field_mask is not None:
            sim_mask = sim_mask * self.field_mask

        # TODO For now, find min and max coords of mask.
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


def vis_field_mask(mask):
    """
    mask: Tensor [H', W'] float 0-1
    """
    vis = mask.cpu().numpy() * 255
    vis = cv2.resize(vis, None, fx=14, fy=14)
    cv2.imshow("Field Mask", vis)
    cv2.waitKey(0)
