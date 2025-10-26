"""
Script that crops and stabilizes video using ffmpeg.
"""

import argparse
import shutil
from pathlib import Path
from subprocess import run

from pointstxt import parse_file

FFMPEG = shutil.which("ffmpeg")
assert FFMPEG is not None


def crop(in_file, out_file, game):
    # Generate ffmpeg trim argument
    trim_arg = ""
    for i, point in enumerate(game.points):
        trim_arg += f"[0:v]trim=start={point.start}:end={point.end},setpts=PTS-STARTPTS[v{i}];"
        trim_arg += f"[0:a]atrim=start={point.start}:end={point.end},asetpts=PTS-STARTPTS[a{i}];"

    end_trim_arg = ""
    for i in range(len(game.points)):
        end_trim_arg += f"[v{i}][a{i}]"
    end_trim_arg += f"concat=n={len(game.points)}:v=1:a=1[outv][outa]"
    trim_arg += end_trim_arg

    ffmpeg_args = [
        FFMPEG,
        "-i", str(in_file),
        "-filter_complex", trim_arg, "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-c:a", "mp3",
        "-s", "1920x1080", "-r", "24", "-crf", "26",
        str(out_file),
    ]
    run(ffmpeg_args, check=True)


def stabilize(in_file, out_file):
    run([
        FFMPEG,
        "-i", str(in_file),
        "-vf", "vidstabdetect=shakiness=6:result=transforms.trf",
        "-f", "null", "-",
    ], check=True)

    run([
        FFMPEG,
        "-i", str(in_file),
        "-vf", "vidstabtransform=input=transforms.trf:smoothing=30:zoom=10",
        "-c:v", "libx264", "-c:a", "copy",
        "-crf", "26",
        str(out_file),
    ], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("points", type=Path)
    args = parser.parse_args()

    game = parse_file(args.points)

    inter_file = args.input.with_suffix(".cropped.mp4")
    crop(args.input, inter_file, game)
    stabilize(inter_file, args.output)


if __name__ == "__main__":
    main()
