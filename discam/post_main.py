"""
Entry point for post-recording video processing.
Crops and trims video.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import torch

from .cv.pipeline import post_run_pipeline
from .post.bounding_box import compute_final_boxes
#from trim import find_trim_sections, gen_timestamps
from .utils.constants import *
from .utils.video_rw import post_write_video

torch.set_grad_enabled(False)


def check_file_exists(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)


def get_file_paths(video_path):
    """
    Get file paths based on input video path.
    """
    def with_suffix(suffix):
        return video_path.parent / (video_path.stem + suffix)
    paths = {
        "in": video_path,
        "out": with_suffix(".discout.mp4"),
        "field_mask": with_suffix(".npy"),
        "cache": with_suffix(".discache"),
    }
    check_file_exists(paths["in"])
    check_file_exists(paths["field_mask"])
    paths["cache"].mkdir(exist_ok=True)
    return paths


def run_pipe_wrapper(args, paths):
    """
    Load from cache, or
    run pipeline and save to cache.
    """
    print("Run CV pipeline:")
    cache_file = paths["cache"] / "pipeline_post.pkl"
    if args.no_cache or not cache_file.exists():
        data = post_run_pipeline(paths["in"], paths["field_mask"])
        print(f"    Saving to cache {cache_file}.")
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    else:
        print(f"    Loading from cache {cache_file}.")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--no_cache", action="store_true", help="Don't read from cache.")
    args = parser.parse_args()

    # Get paths.
    paths = get_file_paths(args.video)
    print(f"Discpost: Discam version {VERSION}",
          f"    Input video: {paths['in']}",
          f"    Output video: {paths['out']}",
          f"    Field mask: {paths['field_mask']}", sep="\n")

    # Get video info.
    cap = cv2.VideoCapture(str(paths["in"]))
    orig_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Input video:",
          f"    Frame count: {orig_len}",
          f"    FPS: {orig_fps}", sep="\n")

    # Run pipeline.
    pipe_outs, frame_is = run_pipe_wrapper(args, paths)

    print("Compute bounding boxes.")
    boxes = compute_final_boxes(pipe_outs, frame_is, orig_len)

    """
    trim_sections = find_trim_sections(detect_out)
    print(f"Found trim sections: ", end="")
    for x in trim_sections:
        print(x, end=" ")
    print()

    ts_string = gen_timestamps(trim_sections)
    ts_file = args.video.parent / (args.video.stem + ".ts.txt")
    print(f"    Writing timestamps to {ts_file}")
    with open(ts_file, "w") as f:
        f.write(ts_string)
    """

    print("Write output video.")
    post_write_video(paths["in"], paths["out"], boxes, None)


if __name__ == "__main__":
    main()
