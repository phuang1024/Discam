"""
Tracking algorithm.
"""

import time
from threading import Thread

import numpy as np

from constants2 import *
from control import PTZControl

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


def control_ptz(ctrl: PTZControl, pan: float, tilt: float, zoom: float):
    """
    Transient thread that executes a single camera movement.

    The camera operates with constant velocity, so:
    The sign of pan, tilt, zoom determines the direction of motion.
    The magnitudes of pan, tilt, zoom determines the duration (seconds) of motion.
    """
    if zoom != 0:
        ctrl.set_zoom(int(np.sign(zoom)))
        time.sleep(abs(zoom))
        ctrl.stop()
    if pan != 0:
        ctrl.set_pt(int(np.sign(pan)), 0)
        time.sleep(abs(pan))
        ctrl.stop()
    if tilt != 0:
        ctrl.set_pt(0, int(np.sign(tilt)))
        time.sleep(abs(tilt))
        ctrl.stop()


def tracking_thread(state: ThreadState):
    """
    Thread that handles tracking players,
    and controlling PTZ camera to follow.
    """
    ptz = PTZControl(PTZ_PATH)

    while state.run:
        # TODO ptz algorithm

        time.sleep(PTZ_INTERVAL)


