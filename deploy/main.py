"""
Deployment script main entry point.

Threading:
- Main thread spawns other threads and manages shared state.
- Recorder thread reads from camera and saves to disk.
- Read camera thread reads camera frames into a queue, maintaining time accuracy.
- PTZ control thread continuously outputs PTZ commands to camera.
  Periodically spawns a transient inference thread.
- Inference thread runs neural network on latest frame, and reports result to PTZ control thread.
  Fetches latest frame from a shared frame buffer with main thread.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

from constants import *
from recorder import camera_read_thread, video_write_thread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()

    # Make timestamped output directory
    out_dir = args.output / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Spawn threads.
    frameq = deque()
    state = ThreadState(frameq=frameq)

    camera_read_t = Thread(target=camera_read_thread, args=(state,))
    video_write_t = Thread(target=video_write_thread, args=(state, out_dir,))

    threads = (
        camera_read_t,
        video_write_t,
    )
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")

    state.run = False
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
