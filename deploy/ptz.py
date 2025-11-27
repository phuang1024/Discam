"""
PTZ control and NN inference.
"""

import time

import numpy as np

from constants2 import *

# DOFs: Pan+, Pan-, Tilt+, Tilt-, Zoom+, Zoom-
# raw_command_i = dot(DOF_ARRAY[i], nn_output)
DOF_ARRAY = np.array([
    [0, 0.5, 0, -0.5],
    [0, -0.5, 0, 0.5],
    [0.5, 0, -0.5, 0],
    [-0.5, 0, 0.5, 0],
    [0.25, 0.25, 0.25, 0.25],
    [-0.25, -0.25, -0.25, -0.25],
])


def ptz_control_thread(state: ThreadState):
    while state.run:
        # TODO ptz algorithm

        time.sleep(PTZ_INTERVAL)


