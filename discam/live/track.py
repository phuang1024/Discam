"""Tracker class.
"""

import numpy as np

from ..utils.constants import *


class Tracker:
    """Algorithm to compute PTZ commands given detected players.
    """

    def __init__(self):
        self.px_per_deg = CV_RES[0] / CAM_FOV

    def update(self, active_boxes, curr_ptz):
        """
        Args:
            active_boxes: ``boxes format``, all active players in coords of current PTZ view.
            curr_ptz: ``ptz format``, current camera position.

        Return:
            ``(pan, tilt, ln(zoom))`` additive delta control.
        """
        if len(active_boxes) == 0:
            # TODO zoom out speed?
            return 0, 0, -1e-2

        pan, tilt, zoom = self.calc_centering_ctrl(active_boxes)
        if len(active_boxes) < TRACK_COUNT:
            zoom -= (TRACK_COUNT - len(active_boxes)) * 1e-2
        print(pan, tilt, zoom)

        pan *= TRACK_PT
        tilt *= TRACK_PT
        zoom *= TRACK_ZOOM_IN if zoom > 0 else TRACK_ZOOM_OUT
        return (pan, tilt, zoom)

    def calc_centering_ctrl(self, active_boxes):
        """Calculate delta PTZ necessary to center detections.
        """
        # Outermost people.
        centers = (active_boxes[:, :2] + active_boxes[:, 2:]) / 2
        x1 = np.min(centers[:, 0])
        x2 = np.max(centers[:, 0])
        y1 = np.min(centers[:, 1])
        y2 = np.max(centers[:, 1])

        # Pan and tilt. Center the outermost people.
        pan_px = (x1 + x2) / 2 - CV_RES[0] / 2
        tilt_px = (y1 + y2) / 2 - CV_RES[1] / 2

        # Zoom.
        target_width = CV_RES[0] - TRACK_MARGIN * 2
        target_height = CV_RES[1] - TRACK_MARGIN * 2
        width_margin = target_width - (x2 - x1)
        height_margin = target_height - (y2 - y1)

        zoom_fac = 1
        if width_margin < height_margin:
            if x2 - x1 > 1e-3:
                zoom_fac = target_width / (x2 - x1)
        else:
            if y2 - y1 > 1e-3:
                zoom_fac = target_height / (y2 - y1)

        pan_ctrl = pan_px / self.px_per_deg
        tilt_ctrl = tilt_px / self.px_per_deg
        zoom_ctrl = np.log(zoom_fac)
        return pan_ctrl, tilt_ctrl, zoom_ctrl
