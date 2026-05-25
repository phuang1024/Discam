"""
Video reader that matches FPS and RES.
"""

import cv2

from utils import RES, FPS


class ScaledReader:
    """
    Video reader with automatic FPS and res scaling.

    Note that FPS scaling will not necessarily be exact.
    I.e. 60fps / 8fps = 7.5, so every 7th or 8th frame will be used.
    """

    def __init__(self, path, fps=FPS, res=RES):
        """
        fps, res: Target FPS and res.
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


if __name__ == "__main__":
    reader = ScaledReader("../data/videos/Irwin.mkv")
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1000 // 8) == ord("q"):
            break
