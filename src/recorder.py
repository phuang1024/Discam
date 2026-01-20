"""
Reader and writer threads.
"""

import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from constants import *

FFMPEG = shutil.which("ffmpeg")
assert FFMPEG is not None


class Recorder:
    def __init__(self, out_dir: Path):
        self.global_frame = 0
        self.chunk_frame = 0
        self.chunk_index = 0

        self.out_dir = out_dir

        # Current video writing.
        self.video = None

        self.start_new_video("0.mp4")

    def write_frame(self, frame: np.ndarray):
        """
        frame: BGR frame to write.
        """
        self.video.write(frame)

        self.global_frame += 1
        self.chunk_frame += 1
        if self.chunk_frame >= RECORD_CHUNK_SIZE:
            self.start_new_chunk()

    def start_new_chunk(self):
        self.chunk_index += 1
        self.chunk_frame = 0
        file_name = f"{self.chunk_index}.mp4"
        self.start_new_video(file_name)

    def stop_video(self):
        if self.video is not None:
            self.video.release()

    def start_new_video(self, file_name: str):
        """
        File is self.out_dir / file_name
        """
        self.stop_video()

        self.curr_file = self.out_dir / file_name
        self.video = cv2.VideoWriter(
            str(self.curr_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (WIDTH, HEIGHT),
        )

        print("Started new VideoWriter:", self.curr_file)

    def close(self):
        self.stop_video()
        print(f"Recorder closed: Recorded {self.global_frame} frames in {self.chunk_index + 1} chunks.")


def writer_thread(state: ThreadState, out_dir):
    """
    Write frames from queue to video files.
    """
    recorder = Recorder(out_dir)

    while state.run:
        # Always leave one frame for the NN.
        if len(state.frameq) <= 1:
            time.sleep(0.01)
            continue

        frame = state.frameq.popleft()
        recorder.write_frame(frame)

    recorder.close()


def reader_thread(state: ThreadState):
    """
    Read camera frames to queue.
    """
    # Warm up
    for _ in range(10):
        state.camera.read()

    while state.run:
        ret, frame = state.camera.read()
        if not ret:
            continue

        state.frameq.append(frame)

        time.sleep(CAM_READ_DELAY)
