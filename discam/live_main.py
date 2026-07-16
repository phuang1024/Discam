"""Entry point for Live mode.
"""

import argparse

from .live.live_pipe import live_run_pipeline
from .utils import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
        help="Path to camera device or video file.")
    parser.add_argument("mask",
        help="Path to mask points.")
    parser.add_argument("--sim", action="store_true",
        help="Simulation mode: Virtual PTZ camera on video.")
    parser.add_argument("--log", action="store_true",
        help="Enable tensorboard logging.")
    args = parser.parse_args()

    if args.log:
        # TODO wrong path for now.
        logger.init_logger(args.mask + ".cache")

    live_run_pipeline(args.input, args.mask, args.sim)
