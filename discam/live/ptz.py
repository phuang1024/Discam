"""PTZ camera and simulated PTZ control API.
"""

import cv2
import numpy as np

from ..utils.constants import *


class PTZ:
    """PTZ interface base class.
    """
    pan: float
    tilt: float
    zoom: float

    def read(self) -> np.ndarray | None:
        """Read next frame.

        Returns:
            ``cv2 format`` frame in CV_RES.
        """
        raise NotImplementedError

    def set_pos(self, pan=None, tilt=None, zoom=None) -> None:
        """Set absolute PTZ position.
        Set arguments to None to keep current pos.
        """
        raise NotImplementedError

    def set_pos_delta(self, pan=None, tilt=None, zoom=None) -> None:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class PTZCamera(PTZ):
    """Control physical PTZ motors.
    """


class PTZSim(PTZ):
    """Simulate PTZ on video with dynamic cropping.
    """

    def __init__(self, path, interval=1):
        """
        Args:
            path: Path to video file.
            interval: Read every Nth frame.
        """
        self.interval = interval
        self.video = cv2.VideoCapture(path)
        self.orig_w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.pan = 0
        self.tilt = 0
        self.zoom = 1

        self.px_per_deg = self.orig_w / CAM_FOV

    def read(self):
        for _ in range(self.interval):
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

        # TODO might go out of bounds.
        frame_crop = self.crop(frame, x1, y1, x1+new_w, y1+new_h)
        frame_crop = cv2.resize(frame_crop, CV_RES)
        return frame_crop

    def crop(self, frame, x1, y1, x2, y2):
        """Clip crop coords to be within frame size.
        """
        x1 = min(max(x1, 0), frame.shape[1])
        x2 = min(max(x2, 0), frame.shape[1])
        y1 = min(max(y1, 0), frame.shape[0])
        y2 = min(max(y2, 0), frame.shape[0])
        return frame[y1:y2, x1:x2]

    def set_pos(self, pan=None, tilt=None, zoom=None):
        if pan is not None:
            self.pan = pan
        if tilt is not None:
            self.tilt = tilt
        if zoom is not None:
            self.zoom = zoom

    def set_pos_delta(self, pan=None, tilt=None, zoom=None):
        if pan is not None:
            self.pan += pan
        if tilt is not None:
            self.tilt += tilt
        if zoom is not None:
            self.zoom += zoom

    def close(self):
        self.video.release()
