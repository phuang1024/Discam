"""Entry point for Live mode.
"""

import argparse
from pathlib import Path

from termcolor import cprint

from .live.live_pipe import live_run_pipeline
from .post_main import check_file_exists
from .utils import logger


def get_file_paths(mask_path):
    """Similar to ``post_main``, but paths are based on mask path.
    """
    def with_suffix(suffix):
        return mask_path.parent / (mask_path.stem + suffix)

    paths = {
        "field_mask": mask_path,
        "cache": with_suffix(".discache"),
    }
    check_file_exists(paths["field_mask"])
    paths["cache"].mkdir(exist_ok=True)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path,
        help="Path to camera device or video file.")
    parser.add_argument("mask", type=Path,
        help="Path to mask points.")
    parser.add_argument("--sim", action="store_true",
        help="Simulation mode: Virtual PTZ camera on video.")
    parser.add_argument("--log", action="store_true",
        help="Enable tensorboard logging.")
    args = parser.parse_args()

    paths = get_file_paths(args.mask)

    if args.log:
        logger.init_logger(paths["cache"])

    live_run_pipeline(args.input, args.mask, args.sim)
