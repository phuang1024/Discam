"""Computer vision sub-pipeline.
"""

import cv2
import numpy as np

from .classify import Classifier
from .detect import Detector
from .perspective import ComputePersp
from ..utils import logger
from ..utils.constants import *


class CVPipeline:
    """CV sub-pipeline.
    Chains the Detector, Perspective, and Classifier together.
    Also handles CV related logging.

    Some modules require an "iteration number", which is kept here.
    """

    def __init__(self, mask_path, tile_size):
        self.detector = Detector(tile_size)
        self.perspective = ComputePersp(mask_path)
        self.classifier = Classifier()

        # CV iteration number.
        self.iter_i = 0

    def update(self, frame, frame_i, ptz=None):
        """
        Args:
            frame: ``cv2 format``.
            frame_i: Index of frame in input video, for logging only.
            ptz: Current PTZ view for live mode.
        """
        # Run modules.
        boxes, adj_boxes = self.detector.update(frame, ptz)

        # Boxes adjusted for PTZ.
        if adj_boxes is None:
            adj_boxes = boxes
        person_locs, mask_locs = self.perspective.update(frame, adj_boxes, self.iter_i)

        active_inds = self.classifier.update(person_locs, mask_locs)

        # Extract return value.
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
            logger.add_scalar("class.sep_metric", self.classifier.sep_metric, frame_i)

        self.iter_i += 1
        return {
            "active_boxes": active_boxes,
            "sep_metric": self.classifier.sep_metric,
        }


def vis_frame(frame, boxes, active_inds):
    """Visualize detected and classified boxes.

    Args:
        frame: ``cv2 format``.
        boxes: ``boxes format``.
        active_inds: ``ndarray bool (N,)``, whether each box is active.
    """
    frame = frame.copy()
    for i, box in enumerate(boxes):
        color = (0, 255, 0) if active_inds[i] else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    return frame


def vis_locations(person_locs, mask_locs, active_inds, persp: ComputePersp, classifier: Classifier):
    """Visualize physical locations.

    Args:
        person_locs: ``ndarray float (N, 2)``, ``locations`` of all detected people.
        mask_locs: ``ndarray float (N, 2)``, ``locations`` of field mask points.
        active_inds: ``ndarray bool (N,)``, whether each person box is active.
    """
    # Image res.
    RES = 800
    # Vis frame side length in "meters".
    GND_SIZE = 140

    def exterp(value, from_min, from_max, to_min, to_max):
        """Linear interp that supports extrap."""
        return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

    def interp_coords(coords):
        """From XY location to vis image pixel pos."""
        return np.array((
            exterp(coords[:, 0], -GND_SIZE // 2, GND_SIZE // 2, 0, RES),
            exterp(coords[:, 1], 0, GND_SIZE, RES, 0),
        ), dtype=int).swapaxes(0, 1)

    img = np.full((RES, RES, 3), 255, dtype=np.uint8)

    # Overall camera FOV cone.
    view_points = np.array(((0, 0), (CV_RES[0], 0), (CV_RES[0], CV_RES[1]), (0, CV_RES[1])), dtype=float)
    view_locs = persp.compute_locations(view_points)
    view_locs = interp_coords(view_locs)
    cv2.fillPoly(img, [view_locs], (200, 200, 200))

    # Classification centers.
    locs = interp_coords(classifier.knn.cluster_centers_)
    for loc in locs:
        cv2.circle(img, loc, 6, (255, 255, 0), 2)

    # Field mask.
    mask_locs = interp_coords(mask_locs)
    cv2.polylines(img, [mask_locs], True, (255, 0, 0), 4)

    # Players.
    person_locs = interp_coords(person_locs)
    for i, loc in enumerate(person_locs):
        color = (0, 255, 0) if active_inds[i] else (0, 0, 255)
        cv2.circle(img, loc, 5, color, -1)

    return img
