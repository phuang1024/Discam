"""
Utilities for incremental reading and recording.
"""

import shutil
import time
from pathlib import Path
from subprocess import Popen, PIPE, DEVNULL

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

        self.ffmpeg = None
        self.curr_file = None

        self.start_new_ffmpeg("0.mp4")

    def write_frame(self, frame: np.ndarray):
        """
        frame: BGR frame to write.
        """
        self.ffmpeg.stdin.write(frame.tobytes())
        self.ffmpeg.stdin.flush()

        self.global_frame += 1
        self.chunk_frame += 1
        if self.chunk_frame >= RECORD_CHUNK_SIZE:
            self.start_new_chunk()

    def start_new_chunk(self):
        self.chunk_index += 1
        self.chunk_frame = 0
        file_name = f"{self.chunk_index}.mp4"
        self.start_new_ffmpeg(file_name)

    def stop_ffmpeg(self):
        if self.ffmpeg is not None:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            if self.ffmpeg.returncode != 0:
                print("Warning: FFmpeg exited with code", self.ffmpeg.returncode, "on file", self.curr_file)

    def start_new_ffmpeg(self, file_name: str):
        """
        File is self.out_dir / file_name
        """
        self.stop_ffmpeg()

        self.curr_file = self.out_dir / file_name
        self.ffmpeg = Popen([
            FFMPEG,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{WIDTH}x{HEIGHT}",
            "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "26",
            str(self.curr_file),
        ], stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)

        print("Started new FFmpeg subprocess:", self.curr_file)

    def close(self):
        self.stop_ffmpeg()
        print(f"Recorder closed: Recorded {self.global_frame} frames in {self.chunk_index + 1} chunks.")


def camera_read_thread(state: ThreadState):
    camera = cv2.VideoCapture(0)

    while state.run:
        ret, frame = camera.read()
        if not ret:
            continue

        state.frameq.append(frame)


def video_write_thread(state: ThreadState, out_dir):
    recorder = Recorder(out_dir)

    while state.run:
        # Always leave one frame for the NN.
        while len(state.frameq) <= 1:
            time.sleep(0.01)

        frame = state.frameq.popleft()
        recorder.write_frame(frame)

    recorder.close()
