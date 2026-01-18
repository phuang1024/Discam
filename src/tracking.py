"""
YOLO detection,
and algorithm to compute camera PTZ movements.
"""

import time

from constants import *


def nn_thread(state: ThreadState):
    while state.run:
        time.sleep(NN_INTERVAL)
        state.nn_output.append((0, 0, 0))
