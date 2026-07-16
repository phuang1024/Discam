"""Live mode main pipeline.
"""

import time

import cv2

from ..cv.pipeline import CVPipeline
from .ptz import PTZCamera, PTZSim


def live_run_pipeline(video_path, mask_path, sim):
    """TODO
    """
    camera = PTZSim(video_path) if sim else PTZCamera(video_path)
    cv_pipe = CVPipeline(mask_path)

    frame_i = 0
    while True:
        frame = camera.read()
        if frame is None:
            break

        cv_pipe.update(frame, frame_i)
        frame_i += 1
