"""
Computer vision visualization functions.
"""

import cv2
import numpy as np

from .perspective import ComputePersp
from ..utils.constants import *


def vis_frame(frame, boxes, active_inds):
    """
    Visualize image frame detections.
    frame: cv2 format.
    boxes: boxes format.
    active_inds: Bool array of whether each box is active.
    """
    frame = frame.copy()
    # Draw boxes.
    for i, box in enumerate(boxes):
        color = (0, 255, 0) if active_inds[i] else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

    cv2.imshow("Pipeline", frame)
    cv2.waitKey(1)


def vis_locations(locs, mask_locs, active_inds, persp: ComputePersp):
    """
    Visualize physical locations.
    """
    RES = 800

    def exterp(value, from_min, from_max, to_min, to_max):
        """Linear interp that supports extrap."""
        return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

    def interp_coords(coords):
        """From XY location to vis image pixel pos."""
        return (
            exterp(coords[:, 0], -50, 50, 0, RES),
            exterp(coords[:, 1], 0, 100, RES, 0),
        )

    img = np.full((RES, RES, 3), 255, dtype=np.uint8)

    # Overall camera FOV cone.
    view_points = np.array(((0, 0), (DET_RES[0], 0), (DET_RES[0], DET_RES[1]), (0, DET_RES[1])), dtype=float)
    view_locs = persp.compute_locations(view_points)
    view_locs = interp_coords(view_locs)
    view_locs = np.array(view_locs, dtype=int).swapaxes(0, 1)
    cv2.fillPoly(img, [view_locs], (200, 200, 200))

    # Field mask.
    mask_locs = interp_coords(mask_locs)
    mask_locs = np.array(mask_locs, dtype=int).swapaxes(0, 1)
    cv2.polylines(img, [mask_locs], True, (255, 0, 0), 4)

    # Players.
    xs, ys = interp_coords(locs)
    for i, (x, y) in enumerate(zip(xs, ys)):
        color = (0, 255, 0) if active_inds[i] else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), 5, color, -1)

    cv2.imshow("Locations", img)
    cv2.waitKey(1)
