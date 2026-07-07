"""Entry point for Post Processing mode.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import torch
from termcolor import cprint

from .post.bounding_box import compute_final_boxes
from .post.run_pipe import post_run_pipeline
from .post.trim import find_trim_sections, gen_timestamps
from .utils import logger
from .utils.constants import *
from .utils.video_rw import post_write_video

torch.set_grad_enabled(False)


def check_file_exists(path):
    if not os.path.exists(path):
        cprint(f"File not found: {path}", "red", attrs=["bold"])
        sys.exit(1)


def get_file_paths(video_path):
    """Get relevant file paths based on input video path.
    """
    def with_suffix(suffix):
        return video_path.parent / (video_path.stem + suffix)

    paths = {
        "in": video_path,
        "out": with_suffix(".discout.mp4"),
        "field_mask": with_suffix(".npy"),
        "cache": with_suffix(".discache"),
        "timestamps": with_suffix(".ts.txt"),
    }
    check_file_exists(paths["in"])
    check_file_exists(paths["field_mask"])
    paths["cache"].mkdir(exist_ok=True)
    return paths


def run_pipe_wrapper(args, paths):
    """Run or load CV pipeline results. Either:

    - Load from cache.
    - Run pipeline and save to cache.
    """
    cprint("Run CV pipeline:", color="light_cyan", attrs=["bold"])
    cache_file = paths["cache"] / "pipeline_post.pkl"

    if args.no_cache or not cache_file.exists():
        # Run pipeline.
        data = post_run_pipeline(paths["in"], paths["field_mask"])
        cprint(f"    Saving to cache {cache_file}.", attrs=["bold"])
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    else:
        # Load.
        cprint(f"    Loading from cache {cache_file}.", attrs=["bold"])
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps_scale", type=int, default=1,
        help="Output FPS downscale factor.")
    parser.add_argument("--trim", action="store_true",
        help="Enable trimming.")
    parser.add_argument("--vcodec", default="libx265",
        help="Video codec (passed to FFmpeg).")
    parser.add_argument("--no_cache", action="store_true",
        help="Don't read from cache.")
    parser.add_argument("--log", action="store_true",
        help="Enable tensorboard logging.")
    args = parser.parse_args()

    # Get paths.
    paths = get_file_paths(args.video)
    cprint(f"Discpost: Discam version {VERSION}", color="light_cyan", attrs=["bold"])
    cprint(f"    Input video: {paths['in']}\n"
           f"    Output video: {paths['out']}\n"
           f"    Field mask: {paths['field_mask']}", attrs=["bold"])

    if args.log:
        cprint(f"Init logger: {paths['cache']}", color="light_cyan", attrs=["bold"])
        logger.init_logger(paths["cache"])

    # Get video info.
    cap = cv2.VideoCapture(str(paths["in"]))
    orig_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    cprint(f"Input video:", color="light_cyan", attrs=["bold"])
    cprint(f"    Frame count: {orig_len}\n"
           f"    FPS: {orig_fps}", attrs=["bold"])

    # Run pipeline.
    pipe_outs, frame_is = run_pipe_wrapper(args, paths)

    # Compute boxes.
    cprint("Compute bounding boxes.", color="light_cyan", attrs=["bold"])
    boxes = compute_final_boxes(pipe_outs, frame_is, orig_len)

    # Find trim sections.
    trim_sections = None
    if args.trim:
        trim_sections = find_trim_sections(pipe_outs)
        cprint(f"Found {len(trim_sections)} trim sections: ", end="", color="light_cyan", attrs=["bold"])
        for x in trim_sections:
            print(x, end=" ")
        print()

        ts_string = gen_timestamps(trim_sections)
        cprint(f"    Writing timestamps to {paths["timestamps"]}", attrs=["bold"])
        with open(paths["timestamps"], "w") as f:
            f.write(ts_string)

    # Write output.
    cprint("Write output video.", color="light_cyan", attrs=["bold"])
    post_write_video(paths["in"], paths["out"], args.fps_scale, boxes, trim_sections, args.vcodec)
