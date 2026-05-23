"""
Video reading and pre-processing utils.
"""

import cv2
import numpy as np

from constants import *


class ScaledReader:
    """
    Video reader with:
    - Automatic FPS and res scaling.
    - Distance warp correction.

    Note that FPS scaling will not necessarily be exact.
    I.e. 60fps / 8fps = 7.5, so every 7th or 8th frame will be used.

    To correct for perspective,
    shrink the lower edge of the image (while keeping upper fixed).
    I.e. make all people appear around the same size.

    Warping is done before resizing.
    """

    def __init__(self, path, fps=FPS, res=RES, warp_fac=0.5):
        """
        fps, res: Target FPS and res.
        warp_fac: Shrink lower edge length by this factor.
            0 = no warp.
        """
        self.cap = cv2.VideoCapture(path)
        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.orig_res = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.new_fps = fps
        self.new_res = res

        # Frame counters in both coordinates.
        self.orig_frame = 0
        self.new_frame = 0

        # Last frame read.
        self.last_frame = None

        # Warp.
        self.warp_fac = warp_fac
        lower_length = self.orig_res[0] * (1 - self.warp_fac)
        rect1 = np.array([
            [0, 0],
            [self.orig_res[0], 0],
            [self.orig_res[0], self.orig_res[1]],
            [0, self.orig_res[1]],
        ], dtype=np.float32)
        rect2 = np.array([
            [0, 0],
            [self.orig_res[0], 0],
            [self.orig_res[0] / 2 + lower_length / 2, self.orig_res[1]],
            [self.orig_res[0] / 2 - lower_length / 2, self.orig_res[1]],
        ], dtype=np.float32)
        self.warp_mat = cv2.getPerspectiveTransform(rect1, rect2)
        self.inverse_warp_mat = cv2.getPerspectiveTransform(rect2, rect1)

    def read(self):
        """
        Returns (success, frame).
        """
        target_frame = self.new_frame * self.orig_fps / self.new_fps
        self.new_frame += 1

        # Read at least one frame.
        if self.last_frame is None:
            ret, self.last_frame = self.cap.read()
            if not ret:
                return False, None
            self.orig_frame += 1

        # Read until target.
        while self.orig_frame + 0.5 < target_frame:
            ret, self.last_frame = self.cap.read()
            if not ret:
                return False, None
            self.orig_frame += 1

        frame = cv2.resize(self.last_frame, self.new_res)
        return True, frame

    def apply_warp(self, frame):
        return cv2.warpPerspective(frame, self.warp_mat, self.orig_res)


if __name__ == "__main__":
    reader = ScaledReader("../data/videos/Irwin.mkv")
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1000 // 8) == ord("q"):
            break
