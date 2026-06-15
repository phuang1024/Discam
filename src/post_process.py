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
import matplotlib.pyplot as plt

from bounding_box import compute_final_boxes
from detect import Detector, vis_detector
from motion import OpticalFlow, vis_of
from utils import *
from video_rw import ScaledReader, FFmpegWriter

torch.set_grad_enabled(False)


def run_of(in_video):
    """
    Run optical flow on video. Convert output time scale to FPS.
    return: ndarray float (T, H, W, 2)
        T: Time. Index i corresponds to index i of the NN.
    """
    video = ScaledReader(in_video, OF_FPS, OF_RES)
    of = OpticalFlow()

    outputs = []
    pbar = tqdm(total=video.get_len(), desc="Optical flow")
    frame_i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        out = of.update(frame)
        vis_of(frame, out)

        # Time scaling.
        if frame_i / OF_FPS >= len(outputs) / FPS:
            outputs.append(out)
            print(frame_i, len(outputs))

        frame_i += 1
        pbar.update(1)

    video.release()
    return np.array(outputs)


def run_detector(in_video, field_mask):
    """
    Run Detector on video.
    return: Sequential list of dict.
        Each dict is a return value from Detector.update
    """
    video = ScaledReader(in_video, FPS, RES)
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
    orig_fps = in_video.get(cv2.CAP_PROP_FPS)
    orig_w = in_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_video = FFmpegWriter(out_path, orig_fps, OUT_RES)

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
        #out_video.write(frame_crop)

        # Draw vis
        vis_frame = frame.copy()
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("box", vis_frame)
        cv2.imshow("crop", frame_crop)
        cv2.waitKey(1)

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
    parser.add_argument("--no_cache", action="store_true", help="Don't load cache.")
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

    check_file_exists(in_path)
    check_file_exists(field_mask_path)

    # Get video info.
    cap = cv2.VideoCapture(args.video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Input video:",
          f"    Frame count: {frame_count}",
          f"    FPS: {out_fps}", sep="\n")

    # Run detector.
    print("Run CV.")
    cache_path = args.video.parent / (args.video.stem + ".discache.pkl")
    if args.no_cache or not cache_path.exists():
        of_out = run_of(in_path)
        stop
        detect_out = run_detector(in_path, field_mask_path)
        print(f"    Saving to cache {cache_path}.")
        with open(cache_path, "wb") as f:
            pickle.dump(detect_out, f)
    else:
        print(f"    Loading from cache {cache_path}.")
        with open(cache_path, "rb") as f:
            detect_out = pickle.load(f)

    # TEST: plot num boxes per frame, over time
    #data = [x["player_count"] for x in detect_out]
    data = [len(x["player_boxes"]) for x in detect_out]
    time = [i / FPS for i in range(len(detect_out))]
    plt.plot(time, data)
    plt.show()
    stop

    print("Compute bounding boxes.")
    boxes = compute_final_boxes(detect_out, frame_count, out_fps)

    print("Write output video.")
    write_output(in_path, out_path, boxes)


if __name__ == "__main__":
    main()
