"""
Deployment script main entry point.

Threading:
- Main thread handles camera.
  Reads frames and adds them to queue.
  Uses results from PTZ thread to control camera.
- Writer thread pops from frame queue and writes to video file.
- PTZ thread periodically uses latest frame to compute PTZ movements.
  Returns result to main thread.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2

from constants import *
from recorder import video_write_thread


def camera_control(state: ThreadState):
    """
    Read camera frames to queue, and apply PTZ movements.
    """
    print("Opening camera:", CAMERA_PATH)
    cap = cv2.VideoCapture(CAMERA_PATH)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Warm up
    for _ in range(10):
        ret, frame = cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        #print("new frame")
        state.frameq.append(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # Make timestamped output directory
    out_dir = args.output / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Spawn threads.
    state = ThreadState(
        frameq=deque(),
        nn_output=deque(maxlen=5),
    )

    video_write_t = Thread(target=video_write_thread, args=(state, out_dir,))

    threads = (
        video_write_t,
    )
    for t in threads:
        t.start()

    try:
        camera_control(state)
    except KeyboardInterrupt:
        print("Stopping threads...")

    state.run = False
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
