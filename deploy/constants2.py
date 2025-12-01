"""
Global constants.

This is named constants2 to avoid conflict with that in train/.
"""

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


# If True, use FakePTZControl and fake_camera_read_thread for testing.
FAKE_TESTING = True

# Path to camera.
#CAMERA_PATH = "/dev/video0"
CAMERA_PATH = "../data/videos/BoomNov10_part.mp4"
# Path to PTZ control serial port.
PTZ_PATH = "/dev/ttyUSB0"

# Recording FPS.
FPS = 24
WIDTH = 1920
HEIGHT = 1080
# Number of frames in each recording chunk.
RECORD_CHUNK_SIZE = 5 * 60 * 24
