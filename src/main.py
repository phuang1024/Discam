import argparse
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2

from camera import PTZCamera, VideoCamera
from constants import *
from control import control_thread
from recorder import reader_thread, writer_thread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # Make timestamped output directory
    out_dir = args.output / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True)
    ctrl_log_dir = out_dir / "control_logs"
    ctrl_log_dir.mkdir()

    # Open camera input.
    #camera = PTZCamera()
    camera = VideoCamera()

    # Spawn threads.
    state = ThreadState(
        camera=camera,
        frameq=deque(),
        nn_output=deque(maxlen=5),
    )

    reader_t = Thread(target=reader_thread, args=(state,))
    writer_t = Thread(target=writer_thread, args=(state, out_dir))
    ctrl_t = Thread(target=control_thread, args=(state, ctrl_log_dir))

    threads = (
        reader_t,
        writer_t,
        ctrl_t,
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
