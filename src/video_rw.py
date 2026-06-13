"""
ScaledReader and FFmpegWriter
"""

import shutil
from subprocess import Popen, PIPE

from utils import *

FFMPEG = shutil.which("ffmpeg")
assert FFMPEG is not None, "ffmpeg not found in PATH"


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

    def get_len(self):
        orig_len = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(orig_len * self.new_fps / self.orig_fps)

    def release(self):
        self.cap.release()


class FFmpegWriter:
    """
    Video writer using ffmpeg subprocess.
    """

    def __init__(self, path, fps=FPS, res=RES):
        self.proc = Popen([
            FFMPEG, "-y",
            "-f", "rawvideo", 
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{res[0]}x{res[1]}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx265",
            "-crf", "28",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            path,
        ], stdin=PIPE, stderr=PIPE, stdout=PIPE)

    def write(self, frame):
        """
        frame: cv2 format.
        """
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()
