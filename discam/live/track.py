"""Tracker class.
"""

import numpy as np

from ..utils.constants import *


class Tracker:
    """Algorithm to compute PTZ commands given detected players.
    """

    def __init__(self):
        pass

    def update(self, active_boxes, curr_ptz):
        """
        Args:
            active_boxes: ``boxes format``, all active players in coords of current PTZ view.
            curr_ptz: ``ptz format``, current camera position.

        Return:
            ``(pan, tilt, zoom)`` additive delta control
                in units defined in ``ptz.py``.
        """
        if len(active_boxes) == 0:
            # TODO zoom out speed?
            return 0, 0, -1e-2

        centers = (active_boxes[:, :2] + active_boxes[:, 2:]) / 2
        # Bounding box around all (i.e. outermost) people.
        x1 = np.min(centers[:, 0])
        x2 = np.max(centers[:, 0])
        y1 = np.min(centers[:, 1])
        y2 = np.max(centers[:, 1])

        # Pan and tilt: Center the outermost people.
        pan_px = (x1 + x2) / 2 - CV_RES[0] / 2
        tilt_px = (y1 + y2) / 2 - CV_RES[1] / 2

        # Zoom: Make width/height of box equal to setpoint.
        width_margin = CV_RES[0] - (x2 - x1) - TRACK_MARGIN * 2
        height_margin = CV_RES[1] - (y2 - y1) - TRACK_MARGIN * 2
        zoom = min(width_margin, height_margin)
        # Account for num detections.
        if len(active_boxes) < TRACK_COUNT:
            # TODO
            zoom -= 1e-2

        pan = pan_px * TRACK_PT_SPEED
        tilt = tilt_px * TRACK_PT_SPEED
        zoom *= TRACK_ZOOM_SPEED
        return pan, tilt, zoom
