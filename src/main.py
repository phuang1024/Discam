import argparse

import torch

from features import *
from transform import solve_transform
from video_reader import VideoReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--keyframe", type=int, default=7)
    args = parser.parse_args()

    video = VideoReader(args.file, 640, args.keyframe)

    while True:
        (key1, des1), (key2, des2), matches = run_orb(video[-1], video[0])

        from_pts = []
        to_pts = []
        for match in matches:
            p1 = key1[match.queryIdx].pt
            p2 = key2[match.trainIdx].pt
            from_pts.append(p1)
            to_pts.append(p2)
        from_pts = torch.tensor(from_pts)
        to_pts = torch.tensor(to_pts)

        trans = solve_transform(from_pts, to_pts)
        print(trans)
        stop

        try:
            video.next()
        except ValueError:
            break


if __name__ == "__main__":
    main()
