"""
Generate NN training data from video and timestamps of active play.
See readme for details.

Clips are fixed length and fixed frame step.

Generated data is:
- Clips as video file.
- Label text file.
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from constants import *


def parse_time(string):
    """
    Convert time string to seconds.
    """
    parts = string.split(":")
    s = 0
    m = 0
    h = 0
    if len(parts) == 1:
        s = int(parts[0])
    elif len(parts) == 2:
        s = int(parts[1])
        m = int(parts[0])
    elif len(parts) == 3:
        s = int(parts[2])
        m = int(parts[1])
        h = int(parts[0])
    return s + 60*m + 3600*h


def read_ts(path) -> list[tuple[float, float]]:
    """
    Read timestamps from file.
    """
    ret = []
    with open(path, "r") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            start = parse_time(parts[0])
            end = parse_time(parts[-1])
            ret.append((start, end))

    return ret


def compute_label(timestamps, frame, fps):
    """
    Compute label at frame.
    If frame is between any of the timestamps, return 1.
    Else, return 0.
    """
    time = frame / fps
    for start, end in timestamps:
        if start <= time <= end:
            return 1
    return 0


def create_video_writer(dir, index, fps):
    """
    Create video writer in output dir.
    """
    path = dir / f"{index}.mp4"
    video = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        VIDEO_RES,
    )
    return video


def write_label(dir, index, label):
    """
    Write label to file in output dir.
    """
    path = dir / f"{index}.label.txt"
    with open(path, "w") as fp:
        fp.write(str(label) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("timestamps", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)

    timestamps = read_ts(args.timestamps)

    video_read = cv2.VideoCapture(args.video)
    fps = video_read.get(cv2.CAP_PROP_FPS)

    # Output clip writer.
    video_write = create_video_writer(args.output, 0, fps)
    write_label(args.output, 0, compute_label(timestamps, 0, fps))

    data_index = 0
    sample_len = 0

    frame_index = 0
    pbar = tqdm(total=int(video_read.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        for _ in range(FRAME_STEP):
            ret, frame = video_read.read()
            frame_index += 1
            pbar.update(1)
        if not ret:
            break

        frame = cv2.resize(frame, VIDEO_RES)
        video_write.write(frame)
        sample_len += 1

        if sample_len == VIDEO_LEN:
            # Start new clip.
            sample_len = 0
            data_index += 1
            video_write.release()
            video_write = create_video_writer(args.output, data_index, fps)
            write_label(args.output, data_index, compute_label(timestamps, frame_index, fps))

    pbar.close()
    video_write.release()


if __name__ == "__main__":
    main()
