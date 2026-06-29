"""
Video read and write utils.
"""

import shutil
from subprocess import Popen, DEVNULL, PIPE

import cv2
from tqdm import tqdm

from .constants import *


class FFmpegWriter:
    """
    Video writer using ffmpeg subprocess.
    """

    def __init__(self, path, fps, res):
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
            "-c:v", "libx265",
            "-crf", "32",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            path,
        ], stdin=PIPE, stderr=DEVNULL, stdout=DEVNULL)

    def write(self, frame):
        """
        frame: cv2 format.
        """
        self.proc.stdin.write(frame.tobytes())
        self.proc.stdin.flush()

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


def post_write_video(in_path, out_path, fps_scale, boxes, trim_sections):
    """
    Write output video in post processing mode.
    Use given bounding boxes to crop.
    Use given trim sections to trim.
    """
    #trim_sections = trim_sections.tolist()

    in_video = cv2.VideoCapture(in_path)
    orig_fps = in_video.get(cv2.CAP_PROP_FPS)
    orig_w = in_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_video = FFmpegWriter(out_path, orig_fps / fps_scale, OUT_RES)

    frame_i = 0
    pbar = tqdm(total=len(boxes), desc="Writing output")
    while True:
        # Increment at beginning.
        frame_i += 1
        pbar.update(1)
        ret, frame = in_video.read()
        if not ret:
            break
        # Check output FPS downscaling.
        if frame_i % fps_scale != 0:
            continue

        # Check trim.
        """
        if len(trim_sections) > 0:
            curr_time = (frame_i - 1) / orig_fps
            if curr_time > trim_sections[0][1]:
                trim_sections.pop(0)
            if trim_sections[0][0] <= curr_time <= trim_sections[0][1]:
                continue
        """

        # Get bbox.
        x1, y1, x2, y2 = boxes[frame_i - 1]
        x1 = int(x1 * orig_w / DET_RES[0])
        y1 = int(y1 * orig_h / DET_RES[1])
        x2 = int(x2 * orig_w / DET_RES[0])
        y2 = int(y2 * orig_h / DET_RES[1])
        # Crop frame
        frame_crop = frame[y1:y2, x1:x2]
        frame_crop = cv2.resize(frame_crop, OUT_RES)
        out_video.write(frame_crop)

        #vis_output_video(frame, frame_crop, (x1, y1, x2, y2))

    pbar.close()
    in_video.release()
    out_video.release()


def vis_output_video(orig_frame, crop_frame, box):
    vis_frame = orig_frame.copy()
    cv2.rectangle(vis_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow("box", vis_frame)
    cv2.imshow("crop", crop_frame)
    cv2.waitKey(1)
