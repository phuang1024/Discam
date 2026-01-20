"""
PTZ control thread.
"""

import time

import cv2
import numpy as np

from constants import *


def control_thread(state: ThreadState):
    # Current PTZ.
    curr_pos = np.array([0, 0, 0], dtype=int)

    while state.run:
        time.sleep(1 / CTRL_FPS)
        if len(state.frameq) == 0:
            continue

        frame = state.frameq[-1]

        cv2.imshow("a", frame)
        cv2.waitKey(1)

        state.camera.set(cv2.CAP_PROP_PAN, curr_pos[0])
        state.camera.set(cv2.CAP_PROP_TILT, curr_pos[1])
        state.camera.set(cv2.CAP_PROP_ZOOM, curr_pos[2])
