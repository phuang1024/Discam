"""
Label frames of video with corresponding transform.
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from features import get_keypoints
from transform import solve_transform, extract_translation_zoom
from video_reader import VideoReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Directory to video files.")
    parser.add_argument("output", type=Path, help="Directory to output frames.")
    parser.add_argument("--keyframe", type=int, default=10)
    parser.add_argument("--res", type=int, default=512)
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True)

    index = 0
    pbar = tqdm()
    for file in args.input.iterdir():
        video = VideoReader(str(file), args.res, args.keyframe)
        while True:
            from_pts, to_pts = get_keypoints(video[-1], video[0])
            trans = solve_transform(from_pts, to_pts)
            translation, zoom = extract_translation_zoom(trans)

            # Save frame with label.
            frame = video[0]
            cv2.imwrite(str(args.output / f"{index}.jpg"), frame)
            with open(args.output / f"{index}.txt", "w") as f:
                f.write(f"{translation[0]} {translation[1]} {zoom}\n")

            try:
                video.next()
                index += 1
            except ValueError:
                break

            pbar.update(1)


if __name__ == "__main__":
    main()
