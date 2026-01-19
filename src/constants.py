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


# Path to camera.
CAMERA_PATH = "/dev/video2"

# Recording FPS.
FPS = 24
WIDTH = 1920
HEIGHT = 1080
# Number of frames in each recording chunk.
RECORD_CHUNK_SIZE = 5 * 60 * 24

# Interval in seconds to run NN.
NN_INTERVAL = 1
