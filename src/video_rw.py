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

        [                   ] major_fps
    ----|--|--|--|----------|--|--|--|----------|--|--|--|----
        [  ] minor_fps     (1  2  3  4) minor_frame_count

    Each read, returns stack of "minor_frame_count" frames.
    """

    def __init__(self, path, res, major_fps, minor_fps, minor_frame_count):
        """
        fps, res: Target FPS and res.
        """
        self.cap = cv2.VideoCapture(path)
        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.orig_res = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.major_fps = major_fps
        self.minor_fps = minor_fps
        self.minor_frame_count = minor_frame_count
        self.res = res

        self.out_frame_i = 0

    def read(self):
        """
        return: ndarray uint8 (T, H, W, 3)
        """
        # Corresponding frame number in input.
        frame_start = int(self.out_frame_i * self.orig_fps / self.major_fps)
        frame_step = int(self.orig_fps / self.minor_fps)
        self.out_frame_i += 1

        frames = np.empty((self.minor_frame_count, self.res[1], self.res[0], 3), dtype=np.uint8)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        for i in range(self.minor_frame_count):
            ret, frame = self.cap.read()
            if not ret:
                return False, None

            frame = cv2.resize(frame, self.res)
            frames[i] = frame
            # Jump forward minor fps.
            if i != self.minor_frame_count - 1:
                for _ in range(frame_step - 1):
                    self.cap.read()

        return True, frames

    def get_len(self):
        orig_len = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(orig_len * self.major_fps / self.orig_fps)

    def release(self):
        self.cap.release()


class FFmpegWriter:
    """
    Video writer using ffmpeg subprocess.
    """

    def __init__(self, path, fps, res):
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
