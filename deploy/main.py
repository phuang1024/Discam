"""
Deployment script main entry point.

Threading:
- Main thread spawns other threads and manages shared state.
- Recorder thread reads from camera and saves to disk.
- Read camera thread reads camera frames into a queue, maintaining time accuracy.
- PTZ control thread continuously outputs PTZ commands to camera.
  Periodically spawns a transient inference thread.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

from constants2 import *
from ptz import ptz_control_thread
from recorder import camera_read_thread, video_write_thread


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

    camera_read_t = Thread(target=camera_read_thread, args=(state,))
    video_write_t = Thread(target=video_write_thread, args=(state, out_dir,))
    ptz_control_t = Thread(target=ptz_control_thread, args=(state,))

    threads = (
        camera_read_t,
        video_write_t,
        ptz_control_t,
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
