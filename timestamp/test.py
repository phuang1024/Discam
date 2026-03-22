"""
Run NN on video and visualize results as timeline.
"""

import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from constants import *
from make_data import read_ts
from model import TsModel


def run_nn(model, nn_input):
    """
    Run NN on a video clip (list of frames) and returns binary label.

    nn_input: List of (H, W, C) uint8 numpy frames.
    """
    x = [torch.from_numpy(frame).float() / 255.0 for frame in nn_input]
    x = torch.stack(x, dim=0).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
    pred = model(x).item()
    cls = int(pred > 0)
    return cls


def infer_video(model, video_path):
    """
    Run NN on entire video (split into clips).
    Returns list of intervals where NN predicts active.
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_index = 0
    nn_input = []
    cls = None
    active_sections = []

    fps = video.get(cv2.CAP_PROP_FPS)
    pbar = tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        ret, frame = video.read()
        pbar.update(1)
        if not ret:
            break

        if frame_index % FRAME_STEP == 0:
            nn_input.append(cv2.resize(frame, VIDEO_RES))

        if len(nn_input) == VIDEO_LEN:
            cls = run_nn(model, nn_input)
            if cls == 1:
                frame_start = frame_index - VIDEO_LEN * FRAME_STEP
                active_sections.append((frame_start / fps, frame_index / fps))
            nn_input = []

        frame_index += 1

    pbar.close()

    return active_sections


def draw_timelines(timestamps):
    # Draw vis image.
    img = np.zeros((50 * len(timestamps), 1000, 3), dtype=np.uint8)
    img[:] = 255

    max_frame = max(max(x[1] for x in ts) for ts in timestamps)

    for i, ts in enumerate(timestamps):
        for start, end in ts:
            start_px = int(start / max_frame * img.shape[1])
            end_px = int(end / max_frame * img.shape[1])
            cv2.rectangle(img, (start_px, i * 50), (end_px, (i + 1) * 50), (0, 255, 0), -1)

    cv2.imwrite("timestamps_vis.png", img)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file.")
    parser.add_argument("model", type=str, help="Path to model file.")
    parser.add_argument("--gt", help="Path to GT timestamps file, if applicable.")
    args = parser.parse_args()

    # Load model.
    model = TsModel().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    # Run inference.
    pred_ts = infer_video(model, args.video)

    # Draw results.
    all_ts = [pred_ts]
    if args.gt is not None:
        gt_ts = read_ts(args.gt)
        all_ts.append(gt_ts)

    draw_timelines(all_ts)


if __name__ == "__main__":
    main()
