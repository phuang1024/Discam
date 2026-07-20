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
        if len(active_boxes) == 0:
            # TODO this should zoom out
            return 0, 0, 0

        centers = (active_boxes[:, :2] + active_boxes[:, 2:]) / 2
        # Distance from each edge to outermost person.
        left_dist = np.min(centers[:, 0]) - TRACK_MARGIN
        right_dist = CV_RES[0] - np.max(centers[:, 0]) - TRACK_MARGIN
        up_dist = np.min(centers[:, 1]) - TRACK_MARGIN
        down_dist = CV_RES[1] - np.max(centers[:, 1]) - TRACK_MARGIN

        pan = (left_dist - right_dist) * TRACK_PT_SPEED
        tilt = (up_dist - down_dist) * TRACK_PT_SPEED
        # TODO
        zoom = min(left_dist + right_dist, up_dist + down_dist) * TRACK_ZOOM_SPEED
        print(left_dist, right_dist, up_dist, down_dist)
        return pan, tilt, zoom
