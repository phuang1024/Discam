"""Tracker class.
"""

import numpy as np

from ..utils.constants import *

# TODO
MARGIN = 100
PT_SCALE = 1e-2
ZOOM_SCALE = 1e-4


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
        mean = np.mean(centers, axis=0)
        xmin = np.min(centers[:, 0])
        xmax = np.max(centers[:, 0])
        ymin = np.min(centers[:, 1])
        ymax = np.max(centers[:, 1])

        left_dist = max(xmin - MARGIN, 0)
        right_dist = max(CV_RES[0] - xmax - MARGIN, 0)
        up_dist = max(ymin - MARGIN, 0)
        down_dist = max(CV_RES[1] - ymax - MARGIN, 0)

        pan = (left_dist - right_dist) * PT_SCALE
        tilt = (up_dist - down_dist) * PT_SCALE
        zoom = min(left_dist + right_dist, up_dist + down_dist) * ZOOM_SCALE
        return pan, tilt, zoom
