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
from tqdm import tqdm

from bounding_box import compute_final_boxes
from detect import Detector, vis_detector
from utils import *
from video_rw import ScaledReader, FFmpegWriter

torch.set_grad_enabled(False)


def run_detector(in_video, field_mask):
    """
    Run Detector on video.
    return: Sequential list of dict.
        Each dict is a return value from Detector.update
    """
    video = ScaledReader(in_video)
    detector = Detector(field_mask)

    outputs = []
    pbar = tqdm(total=video.get_len(), desc="Detector")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        outputs.append(detector.update(frame))
        vis_detector(frame, outputs[-1])
        pbar.update(1)

    video.release()
    return outputs


def write_output(in_path, out_path, bboxes):
    """
    Write output video with bboxes drawn.
    """
    in_video = cv2.VideoCapture(in_path)
    fps = in_video.get(cv2.CAP_PROP_FPS)
    orig_w = in_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_video = FFmpegWriter(out_path, fps, OUT_RES)

    frame_i = 0
    pbar = tqdm(total=len(bboxes), desc="Writing output")
    while True:
        ret, frame = in_video.read()
        if not ret:
            break

        bbox = bboxes[frame_i]
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * orig_w / RES[0])
        x2 = int(x2 * orig_w / RES[0])
        y1 = int(y1 * orig_h / RES[1])
        y2 = int(y2 * orig_h / RES[1])

        # Crop frame
        frame_crop = frame[y1:y2, x1:x2]
        frame_crop = cv2.resize(frame_crop, OUT_RES)
        out_video.write(frame_crop)

        # Draw vis
        """
        vis_frame = frame.copy()
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("box", vis_frame)
        cv2.imshow("crop", frame_crop)
        cv2.waitKey(1)
        """

        frame_i += 1
        pbar.update(1)

    pbar.close()
    in_video.release()
    out_video.release()


def check_file_exists(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", help="If none, is InputFilm.discout.mp4")
    parser.add_argument("--field_mask", help="If none, is InputFilm.npy")
    parser.add_argument("--no_cache", action="store_true", help="Don't load (or write) cache.")
    args = parser.parse_args()

    # Determine file paths.
    in_path = str(args.video)
    if args.output is None:
        out_path = str(args.video.parent / (args.video.stem + ".discout.mp4"))
    else:
        out_path = args.output
    if args.field_mask is None:
        field_mask_path = str(args.video.parent / (args.video.stem + ".npy"))
    else:
        field_mask_path = args.field_mask
    print(f"Discam {VERSION}: Video post processing.",
          f"    Input video: {in_path}",
          f"    Output video: {out_path}",
          f"    Field mask: {field_mask_path}", sep="\n")

    # Check file exists.
    check_file_exists(in_path)
    check_file_exists(field_mask_path)
    """
    if os.path.exists(out_path):
        choice = input(f"Output path {out_path} exists. Overwrite? [y/N] ").strip().lower()
        if choice != "y":
            print("Aborting.")
            sys.exit(1)
    """

    # Get video info.
    cap = cv2.VideoCapture(args.video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Input video:",
          f"    Frame count: {frame_count}",
          f"    FPS: {out_fps}", sep="\n")

    # Run detector.
    print("Run detector.")
    cache_path = args.video.parent / (args.video.stem + ".discache.pkl")
    if args.no_cache or not cache_path.exists():
        pipe_outputs = run_detector(in_path, field_mask_path)
        if not args.no_cache:
            print(f"    Saving to cache {cache_path}.")
            with open(cache_path, "wb") as f:
                pickle.dump(pipe_outputs, f)
    else:
        print(f"    Loading from cache {cache_path}.")
        with open(cache_path, "rb") as f:
            pipe_outputs = pickle.load(f)

    print("Compute bounding boxes.")
    boxes = compute_final_boxes(pipe_outputs, frame_count, out_fps)

    print("Write output video.")
    write_output(in_path, out_path, boxes)


if __name__ == "__main__":
    main()
