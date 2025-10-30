"""
Global constants.
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class ThreadState:
    """
    Data class for synchronizing threads.
    """
    frameq: deque
    run: bool = True


# Recording FPS.
FPS = 24
WIDTH = 640
HEIGHT = 480
# Number of frames in each recording chunk.
RECORD_CHUNK_SIZE = 5 * 24
