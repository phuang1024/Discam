import argparse
import shutil
from collections import namedtuple
from subprocess import Popen, PIPE
from tqdm import tqdm

import cv2

from pointstxt import parse_file, get_fps

Annotation = namedtuple("Annotation", ["start", "end", "text", "x", "y"])

FFMPEG = shutil.which("ffmpeg")
assert FFMPEG is not None

TITLE_TIME = 30 * 3
LINE_TIME = 30 * 5
RESULT_TIME = 30 * 4


def color_and_annotate(args, game):
    """
    Adjust color and annotate.
    """
    # Make a list of annotations.
    annotations = []

    """
    for i, line in enumerate(game.title):
        annotations.append(Annotation(
            start=0,
            end=TITLE_TIME,
            text=line,
            x=50,
            y=100 + i*50
        ))
    """

    score_us = 0
    score_them = 0
    for i, point in enumerate(game.points):
        if point.score == 1:
            score_us += 1
        elif point.score == -1:
            score_them += 1
        else:
            print("Warning: Point has invalid score", point)

        annotations.append(Annotation(
            start=point.start_post_crop,
            end=point.start_post_crop + LINE_TIME,
            text=f"Point {i + 1}",
            x=50,
            y=100,
        ))
        annotations.append(Annotation(
            start=point.start_post_crop,
            end=point.start_post_crop + LINE_TIME,
            text=point.line,
            x=50,
            y=180,
        ))

        end_time = point.start_post_crop + point.end - point.start
        annotations.append(Annotation(
            start=end_time - RESULT_TIME,
            end=end_time,
            text=f"Score: {score_us} - {score_them}",
            x=50,
            y=100
        ))
        annotations.append(Annotation(
            start=end_time - RESULT_TIME,
            end=end_time,
            text=point.result,
            x=50,
            y=180
        ))

    in_video = cv2.VideoCapture(args.input)
    width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(in_video.get(cv2.CAP_PROP_FPS))
    out_video = Popen([
        FFMPEG, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-crf", "24",
        args.output,
    ], stdin=PIPE)

    frame_idx = 0
    #pbar = tqdm(total=int(in_video.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        ret, frame = in_video.read()
        if not ret:
            break
        #pbar.update(1)

        # Annotations this frame.
        curr_anns = []
        for ann in annotations:
            if ann.start <= frame_idx < ann.end:
                curr_anns.append(ann)
            if ann.start > frame_idx:
                break

        # Dim the frame up to the lowest annotation.
        max_y = -1
        for ann in curr_anns:
            if ann.y > max_y:
                max_y = ann.y
        if max_y != -1:
            max_y += 50
            frame[:max_y] = (frame[:max_y] * 0.5).astype(frame.dtype)

        # Draw annotations.
        for ann in curr_anns:
            cv2.putText(frame, ann.text, (ann.x, ann.y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

        out_video.stdin.write(frame.tobytes())
        out_video.stdin.flush()
        if out_video.returncode is not None:
            print("FFMPEG process ended unexpectedly")
            break

        frame_idx += 1

    in_video.release()
    out_video.stdin.close()
    out_video.wait()
    if out_video.returncode != 0:
        print("FFMPEG process ended with error code", out_video.returncode)


def write_chapters(game, file):
    with open(file, "w") as f:
        for i, point in enumerate(game.points):
            # Assume 30 FPS
            seconds = point.start_post_crop // 30
            hrs = seconds // 3600
            mins = seconds // 60
            secs = seconds % 60

            time_str = f"{hrs}:" if hrs > 0 else ""
            time_str += f"{mins}:{secs:02}"

            f.write(f"{time_str} P{i+1}: {point.line}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("points")
    args = parser.parse_args()

    fps = get_fps(args.input)
    game = parse_file(args.points, fps)

    write_chapters(game, args.output + "_chapters.txt")
    color_and_annotate(args, game)


if __name__ == "__main__":
    main()
