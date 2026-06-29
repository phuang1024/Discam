"""
Pipeline running for Post Processing.
"""

import cv2
from tqdm import tqdm

from ..cv.pipeline import CVPipeline
from ..utils.constants import *


def post_run_pipeline(video_path, mask_path):
    """
    Run pipeline on video file.
    Respects FPS and RES constants.
    return: (pipe_out, frame_is)
        pipe_out: List of pipeline outputs.
        frame_is: List of frame indices in original video coord.
    """
    video = cv2.VideoCapture(video_path)
    orig_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = int(video.get(cv2.CAP_PROP_FPS))
    orig_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_scale = int(orig_fps / DET_FPS)

    pipeline = CVPipeline(mask_path)
    pipe_out = []
    frame_is = []
    frame_i = 0
    pbar = tqdm(total=orig_len // fps_scale, desc="Detector")
    while True:
        for _ in range(fps_scale):
            ret, frame = video.read()
        frame_i += fps_scale
        if not ret:
            break

        if orig_w != DET_RES[0] or orig_h != DET_RES[1]:
            frame = cv2.resize(frame, DET_RES)

        pipe_out.append(pipeline.update(frame, frame_i))
        frame_is.append(frame_i)
        pbar.update(1)

    pbar.close()
    video.release()
    return pipe_out, frame_is
