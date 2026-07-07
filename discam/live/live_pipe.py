"""Live mode main pipeline.
"""

import time

import cv2

from .ptz import PTZCamera, PTZSim


def live_run_pipeline(video_path, sim):
    """TODO
    """
    if sim:
        camera = PTZSim(video_path)
    else:
        camera = PTZCamera(video_path)

    # TODO testing PTZ positions.
    def show_next(delay):
        cv2.imshow("test", camera.read())
        cv2.waitKey(int(delay * 1000))

    show_next(1)
    for _ in range(5):
        camera.set_pos(zoom=camera.zoom + 0.1)
        show_next(0.3)
    for _ in range(5):
        camera.set_pos(pan=camera.pan + 2)
        show_next(0.3)
    for _ in range(5):
        camera.set_pos(tilt=camera.tilt + 2)
        show_next(0.3)
