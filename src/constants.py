from collections import deque
from dataclasses import dataclass

import cv2


@dataclass
class ThreadState:
    """
    Data class for communication between threads.
    """
    # Input camera.
    camera: cv2.VideoCapture
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
CAMERA_PATH = "../data/videos/RiseNotre_part.mp4"
# If reading from a file, set this nonzero.
CAM_READ_DELAY = 1 / FPS

# FPS to run PTZ control algorithm.
CTRL_FPS = 3
# For all tracking parameters (e.g. bbox size, pos), average over this many control frames.
CTRL_AVG_WINDOW = 10
# Detection confidence threshold.
CONF_THRES = 0.1
