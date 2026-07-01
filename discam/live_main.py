"""Entry point for Live mode.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
        help="Path to camera device or video file.")
    parser.add_argument("output",
        help="Path to write output video.")
    parser.add_argument("--sim", action="store_true",
        help="Simulation mode: Virtual PTZ camera on video.")
