"""
Computer vision sub-pipeline.
"""

import cv2
import numpy as np

from .classify import Classifier
from .detect import Detector
from .perspective import ComputePersp
from ..utils import logger
from ..utils.constants import *


class CVPipeline:
    def __init__(self, mask_path):
        self.detector = Detector()
        self.perspective = ComputePersp(mask_path)
        self.classifier = Classifier()

        # For logging.
        self.iter_i = 0

    def update(self, frame, frame_i):
        """
        frame: cv2 format.
        frame_i: Index of frame.
        """
        boxes = self.detector.update(frame)
        person_locs, mask_locs = self.perspective.update(boxes, self.iter_i)
        active_inds = self.classifier.update(person_locs, mask_locs)
        active_boxes = boxes[active_inds]

        # Vis and logging.
        if logger.enabled:
            det_vis = vis_frame(frame, boxes, active_inds)
            locs_vis = vis_locations(person_locs, mask_locs, active_inds, self.perspective, self.classifier)
            if self.iter_i % LOG_IMG_INTERVAL == 0:
                logger.add_image("vis.detections", det_vis, frame_i)
                logger.add_image("vis.locations", locs_vis, frame_i)
            cv2.imshow("Detections", det_vis)
            cv2.imshow("Locations", locs_vis)
            cv2.waitKey(1)

            logger.add_scalar("persp.vanishing", self.perspective.vanishing, frame_i)
            logger.add_scalar("persp.min_dist", self.perspective.dist_min, frame_i)

            logger.add_scalar("class.num_active", np.sum(active_inds), frame_i)
            logger.add_scalar("class.gmm1_bic", self.classifier.gmm1_bic, frame_i)
            logger.add_scalar("class.gmm2_bic", self.classifier.gmm2_bic, frame_i)

        self.iter_i += 1
        return {
            "active_boxes": active_boxes,
        }


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

    return frame


def vis_locations(locs, mask_locs, active_inds, persp: ComputePersp, classifier: Classifier):
    """
    Visualize physical locations.
    """
    RES = 800

    def exterp(value, from_min, from_max, to_min, to_max):
        """Linear interp that supports extrap."""
        return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

    def interp_coords(coords):
        """From XY location to vis image pixel pos."""
        return np.array((
            exterp(coords[:, 0], -50, 50, 0, RES),
            exterp(coords[:, 1], 0, 100, RES, 0),
        ), dtype=int)

    img = np.full((RES, RES, 3), 255, dtype=np.uint8)

    # Overall camera FOV cone.
    view_points = np.array(((0, 0), (DET_RES[0], 0), (DET_RES[0], DET_RES[1]), (0, DET_RES[1])), dtype=float)
    view_locs = persp.compute_locations(view_points)
    view_locs = interp_coords(view_locs).swapaxes(0, 1)
    cv2.fillPoly(img, [view_locs], (200, 200, 200))

    # Classification Gaussians.
    """
    for (mean, cov) in zip(classifier.gmm.means_, classifier.gmm.covariances_):
        # For simplicity, just draw a quad.
        eigvals, eigvects = np.linalg.eig(cov)
        axes = eigvals * eigvects
        points = np.array((
            mean + axes[0],
            mean + axes[1],
            mean - axes[0],
            mean - axes[1],
        ), dtype=float)
        points = interp_coords(points).swapaxes(0, 1)
        cv2.fillPoly(img, [points], (200, 160, 160))
    """

    # Field mask.
    mask_locs = interp_coords(mask_locs).swapaxes(0, 1)
    cv2.polylines(img, [mask_locs], True, (255, 0, 0), 4)

    # Players.
    xs, ys = interp_coords(locs)
    for i, (x, y) in enumerate(zip(xs, ys)):
        color = (0, 255, 0) if active_inds[i] else (0, 0, 255)
        cv2.circle(img, (x, y), 5, color, -1)

    return img
