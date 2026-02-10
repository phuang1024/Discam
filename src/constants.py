from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from camera import BaseCamera


@dataclass
class ThreadState:
    """
    Data class for communication between threads.
    """
    # Input camera.
    camera: BaseCamera
    # Queue of frames read from camera. Guaranteed to contain at least one element (except at init).
    frameq: deque
    # Queue of neural network outputs. Set a max length.
    nn_output: deque
    run: bool = True


# Recording FPS.
FPS = 24
WIDTH = 1920
HEIGHT = 1080
# Number of frames in each recording chunk.
RECORD_CHUNK_SIZE = 5 * 60 * 24

# Path to camera.
#CAMERA_PATH = "/dev/video2"
CAMERA_PATH = "../data/videos/BoomJan29_part.mp4"

# FPS to run PTZ control algorithm.
CTRL_FPS = 2
# Delay after sending control.
CTRL_DELAY = 3
# Speeds and thresholds of axes.
PT_SPEED = 50 * 3600
PT_THRES = 0.1
ZOOM_CENTER = 0.5
ZOOM_THRES = 0.1
ZOOM_SPEED = 30
ZOOM_MAX = 30
# Moving average length for bbox detections.
CTRL_AVG_WINDOW = 5
# Detection confidence threshold.
CONF_THRES = 0.1
# Quantile of all detection positions to use for median box.
BOX_QUANTILE = 0.1
