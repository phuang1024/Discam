import argparse

from features import *
from video_reader import VideoReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--keyframe", type=int, default=7)
    args = parser.parse_args()

    video = VideoReader(args.file, args.keyframe)


if __name__ == "__main__":
    main()
