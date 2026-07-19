"""Tracker class.
"""

import numpy as np

from ..utils.constants import *


class Tracker:
    """Algorithm to compute PTZ commands given detected players.
    """

    def __init__(self):
        pass

    def update(self, active_boxes):
        """
        TODO should probably take current PTZ pos too.
        Args:
            active_boxes: ``boxes format``, all active players in coords of current PTZ view.

        Return:
            ``(pan, tilt, zoom)`` additive delta control
                in units defined in ``ptz.py``.
        """
        # TODO
        return self.compute_track(active_boxes)

    def compute_track(self, active_boxes):
        """Compute delta PTZ values as "logits".
        Magnitude 1 is "standard" speed.
        """
        if len(active_boxes) == 0:
            # TODO this should zoom out
            return 0, 0, 0

        centers = (active_boxes[:, :2] + active_boxes[:, 2:]) / 2
        xmin = np.min(centers[:, 0])
        xmax = np.max(centers[:, 0])
        ymin = np.min(centers[:, 1])
        ymax = np.max(centers[:, 1])

        left_dist = max(xmin - TRACK_MARGIN, 0)
        right_dist = max(CV_RES[0] - xmax - TRACK_MARGIN, 0)
        up_dist = max(ymin - TRACK_MARGIN, 0)
        down_dist = max(CV_RES[1] - ymax - TRACK_MARGIN, 0)

        pan = (left_dist - right_dist) * TRACK_PT_SPEED
        tilt = (up_dist - down_dist) * TRACK_PT_SPEED
        zoom = min(left_dist + right_dist, up_dist + down_dist) * TRACK_ZOOM_SPEED
        return pan, tilt, zoom
