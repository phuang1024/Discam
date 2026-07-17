"""Live mode main pipeline.
"""

import time

import cv2

from ..cv.pipeline import CVPipeline
from .ptz import PTZCamera, PTZSim
from .track import Tracker


def live_run_pipeline(video_path, mask_path, sim):
    """TODO
    """
    camera = PTZSim(video_path) if sim else PTZCamera(video_path)
    cv_pipe = CVPipeline(mask_path)
    tracker = Tracker()

    frame_i = 0
    while True:
        frame = camera.read()
        if frame is None:
            break

        pipe_out = cv_pipe.update(frame, frame_i)
        frame_i += 1

        delta_ptz = tracker.update(pipe_out["active_boxes"])
        camera.set_pos_delta(*delta_ptz)
