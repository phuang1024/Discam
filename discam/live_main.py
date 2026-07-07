"""Entry point for Live mode.
"""

import argparse

from .live.live_pipe import live_run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
        help="Path to camera device or video file.")
    parser.add_argument("--sim", action="store_true",
        help="Simulation mode: Virtual PTZ camera on video.")
    args = parser.parse_args()

    live_run_pipeline(args.input, args.sim)
