"""
Run this script after recording to put all the clips into a single video.
"""

import argparse
import os
import shutil
from pathlib import Path
from subprocess import run

FFMPEG = shutil.which("ffmpeg")
assert FFMPEG is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    files = sorted(list(args.input.iterdir()))

    ff_args = [FFMPEG]
    for f in files:
        ff_args.append("-i")
        ff_args.append(str(f))

    ff_args.append("-filter_complex")
    filter_str = ""
    for i in range(len(files)):
        filter_str += f"[{i}:v:0]"
    filter_str += f"concat=n={len(files)}:v=1:a=0[outv]"
    ff_args.append(filter_str)

    ff_args.append("-map")
    ff_args.append("[outv]")
    ff_args.append(str(args.output))

    run(ff_args, check=True)


if __name__ == "__main__":
    main()
