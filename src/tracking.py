"""
Tracking algorithm.

Uses YOLO tracking and other video processing
to determine and periodically execute movements on the camera.
"""

import time
from threading import Thread

import numpy as np

from constants2 import *
from control import PTZControl, FakePTZControl
from detection import YoloTracker, compute_track_speed


def control_ptz(ctrl: PTZControl, pan: float, tilt: float, zoom: float):
    """
    Executes a single camera movement.

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
    if FAKE_TESTING:
        ptz = FakePTZControl()
    else:
        ptz = PTZControl(PTZ_PATH)

    yolo = YoloTracker()

    yolo_t = Thread(target=yolo_thread, args=(state, yolo))
    yolo_t.start()

    while state.run:
        yolo.clear_tracks()
        time.sleep(10)

        # Tracks with sufficient speed.
        speedy_players = []
        for track in yolo.tracks.values():
            spd = compute_track_speed(track)
            if spd > TRACK_SPEED_THRES:
                speedy_players.append(track)
        print(f"Found {len(speedy_players)} speedy players")
        if len(speedy_players) < TRACK_COUNT_THRES:
            continue

        # Compute bounding box around speedy players.
        min_x = WIDTH
        max_x = 0
        min_y = HEIGHT
        max_y = 0
        for track in speedy_players:
            x, y = track[-1]
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

        box_cx = (min_x + max_x) / 2
        box_cy = (min_y + max_y) / 2

        # Check if box is far enough off center.
        delta_x = box_cx - WIDTH / 2
        delta_y = box_cy - HEIGHT / 2
        print(f"Box center offset: ({delta_x}, {delta_y})")
        if abs(delta_x) < TRACK_PIXELS_THRES and abs(delta_y) < TRACK_PIXELS_THRES:
            continue

        # Compute pan/tilt durations.
        pan_dur = delta_x / WIDTH * 0.01
        tilt_dur = -delta_y / HEIGHT * 0.01
        # Execute movement.
        control_ptz(ptz, pan_dur, tilt_dur, 0)
