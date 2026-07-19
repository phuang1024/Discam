"""Video write utils.
"""

import shutil
from subprocess import Popen, DEVNULL, PIPE

import cv2
from tqdm import tqdm

from .constants import *


class FFmpegWriter:
    """Video writer using ffmpeg subprocess.
    """

    def __init__(self, path, fps, res, vcodec):
        ffmpeg = shutil.which("ffmpeg")
        assert ffmpeg is not None, "ffmpeg not found in PATH"

        self.proc = Popen([
            ffmpeg, "-y",
            "-f", "rawvideo", 
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{res[0]}x{res[1]}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", vcodec,
            "-crf", "32",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            path,
        ], stdin=PIPE, stderr=DEVNULL, stdout=DEVNULL)

    def write(self, frame):
        """
        Args:
            frame: ``cv2 format``.
        """
        self.proc.stdin.write(frame.tobytes())
        self.proc.stdin.flush()

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


def post_write_video(in_path, out_path, fps_scale, boxes, trim_sections, vcodec):
    """Write output video in Post Processing mode, given crop boxes.
    Also trims video.

    Args:
        fps_scale: ``out_fps = in_fps / scale``, int.
        boxes: ``boxes format``, crop boxes for each frame.
        trim_sections: ``ndarray float (N, 2)``, sections (sec) to remove.
            Should be sorted.
            Can be ``None`` for no trimming.
    """
    in_video = cv2.VideoCapture(in_path)
    orig_fps = in_video.get(cv2.CAP_PROP_FPS)
    orig_w = in_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_video = FFmpegWriter(out_path, orig_fps / fps_scale, OUT_RES, vcodec)

    if trim_sections is not None:
        # Convert sec to frames. Will be popped incrementally.
        trim_sections = (trim_sections * orig_fps).astype(int).tolist()

    # Some values are skipped.
    frame_i = 0
    pbar = tqdm(total=len(boxes), desc="Writing output")
    while True:
        ret, frame = in_video.read()
        if not ret:
            break

        # Check trim.
        if trim_sections and frame_i > trim_sections[0][0]:
            in_video.set(cv2.CAP_PROP_POS_FRAMES, trim_sections[0][1])
            frame_i = trim_sections[0][1]
            trim_sections.pop(0)
            continue

        # Get bbox.
        x1, y1, x2, y2 = boxes[frame_i - 1]
        x1 = int(x1 * orig_w / CV_RES[0])
        y1 = int(y1 * orig_h / CV_RES[1])
        x2 = int(x2 * orig_w / CV_RES[0])
        y2 = int(y2 * orig_h / CV_RES[1])
        # Crop frame
        frame_crop = frame[y1:y2, x1:x2]
        frame_crop = cv2.resize(frame_crop, OUT_RES)
        out_video.write(frame_crop)
        #vis_output_video(frame, frame_crop, (x1, y1, x2, y2))

        frame_i += fps_scale
        pbar.n = frame_i
        pbar.refresh()

    pbar.close()
    in_video.release()
    out_video.release()


def vis_output_video(orig_frame, crop_frame, box):
    """Visualize output video frames as it's being written.
    """
    vis_frame = orig_frame.copy()
    cv2.rectangle(vis_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow("Crop box", vis_frame)
    cv2.imshow("Output frame", crop_frame)
    cv2.waitKey(1)
