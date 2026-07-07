"""PTZ camera and simulated PTZ control API.
"""

import cv2
import numpy as np

from ..utils.constants import *


class PTZ:
    """PTZ interface base class.
    """

    def read(self) -> np.ndarray | None:
        """Read next frame.

        Returns:
            ``cv2 format`` frame in any resolution.
        """
        raise NotImplementedError

    def set_pos(self, pan=None, tilt=None, zoom=None) -> None:
        """Set absolute PTZ position.
        Set arguments to None to keep current pos.
        """
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class PTZCamera(PTZ):
    """Control physical PTZ motors.
    """


class PTZSim(PTZ):
    """Simulate PTZ on video with dynamic cropping.

    Pan and tilt is degrees off of default video view.
    Zoom is factor > 1.
    """

    def __init__(self, path):
        """
        Args:
            path: Path to video file.
        """
        self.video = cv2.VideoCapture(path)
        self.orig_w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.pan = 0
        self.tilt = 0
        self.zoom = 1

        self.px_per_deg = self.orig_w / CAM_FOV

    def read(self):
        ret, frame = self.video.read()
        if not ret:
            return None

        # Crop frame.
        new_w = int(self.orig_w / self.zoom)
        new_h = int(self.orig_h / self.zoom)
        offset_x = self.px_per_deg * self.pan
        offset_y = self.px_per_deg * self.tilt
        x1 = int(self.orig_w // 2 + offset_x - new_w // 2)
        y1 = int(self.orig_h // 2 + offset_y - new_h // 2)
        print("Crop xywh:", x1, y1, new_w, new_h)

        # TODO might go out of bounds.
        frame_crop = frame[y1 : y1+new_h, x1 : x1+new_w]
        #frame_crop = cv2.resize(frame_crop, CV_RES)
        return frame_crop

    def set_pos(self, pan=None, tilt=None, zoom=None):
        if pan is not None:
            self.pan = pan
        if tilt is not None:
            self.tilt = tilt
        if zoom is not None:
            self.zoom = zoom

    def close(self):
        self.video.release()
