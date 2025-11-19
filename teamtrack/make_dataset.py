"""
Process TeamTrack dataset into the format I want.

NOTE: Currently, this is written to match the format in "basketball" directories.
"""

import argparse
from pathlib import Path

import cv2

# Dimensions of frames to write.
WIDTH = 1920
HEIGHT = 1080
# Scaling from input data to output data
# I.e. 0.5 means the original data was twice the resolution.
SCALING = 0.5


def generate_preview(frame, bboxes: list[tuple]):
    """
    Draw the bboxes on the frame.
    """
    frame = frame.copy()
    for bbox in bboxes:
        x, y, w, h = bbox
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
        # Read frame.
        for _ in range(frame_skip):
            ret, frame = video.read()
            if not ret:
                return num_written
            video_index += 1

        # Write frame.
        frame = cv2.resize(frame, (WIDTH, HEIGHT), cv2.INTER_AREA)
        out_frame_f = out_dir / f"{out_index}.jpg"
        cv2.imwrite(str(out_frame_f), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        frame_bboxes = bboxes[video_index - 1]

        """
        # Write preview with bboxes.
        preview = generate_preview(frame, frame_bboxes)
        preview = cv2.resize(preview, (WIDTH // 2, HEIGHT // 2), cv2.INTER_AREA)
        out_preview_f = out_dir / f"{out_index}_preview.jpg"
        cv2.imwrite(str(out_preview_f), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        """

        # Write annotation.
        out_anno_f = out_dir / f"{out_index}.txt"
        for bbox in frame_bboxes:
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            norm_cx = cx / WIDTH
            norm_cy = cy / HEIGHT
            norm_w = w / WIDTH
            norm_h = h / HEIGHT
            with open(out_anno_f, "a") as f:
                f.write(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}\n")

        num_written += 1
        out_index += 1


def process_data(args, files):
    out_index = 0

    for video_f, anno_f in files:
        print("Processing", video_f.name)

        try:
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
                        x *= SCALING
                        y *= SCALING
                        w *= SCALING
                        h *= SCALING
                        bboxes[-1].append((x, y, w, h))

            out_index += process_video(
                args.output,
                out_index,
                video_f,
                bboxes,
                args.frame_skip,
            )

        except Exception as e:
            print("Error processing", video_f.name, ":", e)


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
