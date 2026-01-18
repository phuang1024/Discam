from collections import deque
from dataclasses import dataclass


@dataclass
class ThreadState:
    """
    Data class for communication between threads.
    """
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
RECORD_CHUNK_SIZE = 120#5 * 60 * 24
