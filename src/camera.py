"""
This file defines a camera base class.
The class contains attributes such as read frame, set PTZ.

One implementation is the actual PTZ camera.
Another is a simulated PTZ bounding box, for a video file.
"""

import time

import cv2
import numpy as np

from constants import *


class BaseCamera:
    """
    Camera base class.

    Use CAMERA_PATH as camera or video path.

    pan, tilt are in arcseconds with zero facing forwards.
    90 deg = 324k arcsec.
    zoom is 1 to 100.
    """

    def __init__(self):
        raise NotImplementedError

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read next frame as (H, W, 3) uint8 BGR image.

        return: (success, frame)
        """
        raise NotImplementedError

    def set_ptz(self, pan: int | None, tilt: int | None, zoom: int | None):
        """
        Set PTZ position.
        None means no change.
        """
        raise NotImplementedError


class PTZCamera(BaseCamera):
    """
    PTZ camera. Stream and control via USB.
    """

    def __init__(self):
        print("Opening camera:", CAMERA_PATH)
        self.cap = cv2.VideoCapture(CAMERA_PATH)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

    def read(self):
        return self.cap.read()

    def set_ptz(self, pan, tilt, zoom):
        if pan is not None:
            self.cap.set(cv2.CAP_PROP_PAN, pan)
        if tilt is not None:
            self.cap.set(cv2.CAP_PROP_TILT, tilt)
        if zoom is not None:
            self.cap.set(cv2.CAP_PROP_ZOOM, zoom)


class VideoCamera(BaseCamera):
    """
    Read from a video file, and simulate PTZ with cropping.

    PTZ crop:
    Crop width/height factor is linear with zoom value.
    Assume FOV.
    Bound pan/tilt at image bounds.
    """

    def __init__(self):
        print("Opening video as camera:", CAMERA_PATH)
        self.cap = cv2.VideoCapture(CAMERA_PATH)

        self.fov = 90

        self.pan = 0
        self.tilt = 0
        self.zoom = 0

    def read(self):
        time.sleep(1 / FPS)

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Apply PTZ.
        scale = np.interp(self.zoom, [1, 100], [1, 0.1])
        width = int(scale * WIDTH)
        height = int(scale * HEIGHT)
        max_pan = (WIDTH - width) // 2
        max_tilt = (HEIGHT - height) // 2

        # Convert self.pan and tilt from arcseconds to pixels.
        pan = self.pan / 3600 * WIDTH / self.fov
        tilt = -self.tilt / 3600 * WIDTH / self.fov
        # Constrain pan tilt.
        pan = max(min(pan, max_pan), -max_pan)
        tilt = max(min(tilt, max_tilt), -max_tilt)
        # Add offset.
        pan = int(pan) + WIDTH // 2
        tilt = int(tilt) + HEIGHT // 2

        frame = frame[
            tilt - height // 2 : tilt + height // 2,
            pan - width // 2 : pan + width // 2,
        ]
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        return True, frame

    def set_ptz(self, pan, tilt, zoom):
        if pan is not None:
            self.pan = pan
        if tilt is not None:
            self.tilt = tilt
        if zoom is not None:
            self.zoom = zoom


if __name__ == "__main__":
    cam = VideoCamera()

    cam.zoom = 30

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow("a", frame)
        cv2.waitKey(1)

        cam.tilt += 1000
