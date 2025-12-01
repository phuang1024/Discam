"""
Tracking algorithm.

Uses YOLO tracking and other video processing
to determine and periodically execute movements on the camera.
"""

import time
from threading import Thread

import numpy as np

from constants2 import *
from control import PTZControl
from detection import YoloTracker, compute_track_speed


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


def yolo_thread(state: ThreadState, yolo: YoloTracker):
    """
    Continuously run yolo tracker on latest frame.
    Note: The frame interval thus depends on computer and camera speed.
    """
    while state.run:
        if len(state.frameq) > 0:
            yolo.step(state.frameq[-1])
        time.sleep(0.01)


def tracking_thread(state: ThreadState):
    """
    Thread that handles tracking players,
    and controlling PTZ camera to follow.
    """
    #ptz = PTZControl(PTZ_PATH)
    yolo = YoloTracker()

    yolo_t = Thread(target=yolo_thread, args=(state, yolo))
    yolo_t.start()

    while state.run:
        yolo.clear_tracks()
        time.sleep(10)

        for track in yolo.tracks.values():
            spd = compute_track_speed(track)
            print(spd)
