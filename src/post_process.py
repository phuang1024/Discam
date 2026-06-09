"""
Entry point for post-recording video processing.
Crops and trims video.
"""

import argparse
import pickle

import cv2
import torch
from tqdm import tqdm

from bounding_box import compute_final_boxes
from detect import Detector
from video_read import ScaledReader
from utils import *

torch.set_grad_enabled(False)


def run_detector(args):
    """
    Run Detector on video.
    return: Sequential list of dict.
        Each dict is a return value from Detector.update
    """
    video = ScaledReader(args.video)
    detector = Detector(args.field_mask)

    outputs = []
    pbar = tqdm(total=video.get_len(), desc="CV pipeline")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        outputs.append(detector.update(frame))
        pbar.update(1)

    video.release()
    return outputs


def write_output(args, bboxes):
    """
    Write output video with bboxes drawn.
    """
    in_video = cv2.VideoCapture(args.video)
    fps = in_video.get(cv2.CAP_PROP_FPS)
    orig_w = in_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, OUT_RES)

    frame_i = 0
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
        vis_frame = frame.copy()
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("box", vis_frame)
        cv2.imshow("crop", frame_crop)
        cv2.waitKey(1)

        frame_i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--field_mask")
    args = parser.parse_args()

    # Get video info.
    cap = cv2.VideoCapture(args.video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    """
    pipe_outputs = run_detector(args)
    with open("pipe_out.pkl", "wb") as f:
        pickle.dump(pipe_outputs, f)
    stop
    """
    with open("pipe_out.pkl", "rb") as f:
        pipe_outputs = pickle.load(f)

    boxes = compute_final_boxes(pipe_outputs, frame_count, out_fps)
    write_output(args, boxes)


if __name__ == "__main__":
    main()
