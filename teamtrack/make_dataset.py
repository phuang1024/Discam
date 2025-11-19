"""
Process TeamTrack dataset into the format I want.

NOTE: Currently, this is written to match the format in "basketball" directories.
"""

import argparse
import json
from pathlib import Path

import cv2


def generate_preview(frame, bboxes: list[tuple], scaling=0.5):
    """
    Draw the bboxes on the frame.
    """
    frame = frame.copy()
    for bbox in bboxes:
        x, y, w, h = bbox
        x = x * scaling
        y = y * scaling
        w = w * scaling
        h = h * scaling
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def process_video(out_dir, out_index, video_f, bboxes: list[list[tuple]], frame_skip):
    """
    out_dir: Output directory.
    out_index: Index to begin naming output files.
    video_f: Path to video file.
    bboxes: List of (x, y, w, h) tuples for each frame.
    return: Number of frames written.
    """
    video = cv2.VideoCapture(str(video_f))

    video_index = 0
    num_written = 0
    while True:
        for _ in range(frame_skip):
            ret, frame = video.read()
            if not ret:
                return num_written
            video_index += 1

        frame = cv2.resize(frame, (1920, 1080), cv2.INTER_AREA)
        out_frame_f = out_dir / f"{out_index}.jpg"
        cv2.imwrite(str(out_frame_f), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        frame_bboxes = bboxes[video_index - 1]

        preview = generate_preview(frame, frame_bboxes)
        preview = cv2.resize(preview, (960, 540), cv2.INTER_AREA)
        out_preview_f = out_dir / f"{out_index}_preview.jpg"
        cv2.imwrite(str(out_preview_f), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        # TODO anno

        num_written += 1
        out_index += 1


def process_data(args, files):
    out_index = 0

    for video_f, anno_f in files:
        print("Processing", video_f.name)

        # Read annotations.
        bboxes = []
        with open(anno_f, "r") as f:
            # Skip header lines.
            lines = f.readlines()[4:]
            for line in lines:
                bboxes.append([])
                parts = line.strip().split(",")
                parts = list(map(float, parts[1:]))
                for i in range(0, len(parts), 4):
                    h, x, y, w = parts[i : i + 4]
                    bboxes[-1].append((x, y, w, h))

        out_index += process_video(
            args.output,
            out_index,
            video_f,
            bboxes,
            args.frame_skip,
        )

        stop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="E.g. teamtrack/basketball_side/train")
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame_skip", type=int, default=5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Find video-annotation pairs.
    video_dir = args.input / "videos"
    anno_dir = args.input / "annotations"
    files = []
    for video_f in video_dir.glob("*.mp4"):
        anno_f = anno_dir / f"{video_f.stem}.csv"
        if anno_f.exists():
            files.append((video_f, anno_f))

    process_data(args, files)


if __name__ == "__main__":
    main()
