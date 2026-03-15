"""
Play video and continuously run NN.
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
    nn_input = [torch.from_numpy(frame).float() / 255.0 for frame in nn_input]
    x = torch.stack(nn_input, dim=0).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
    pred = model(x).item()
    cls = int(pred > 0)
    return cls


def draw_frame(frame, cls):
    if cls is not None:
        color = (0, 255, 0) if cls == 1 else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to video file.")
    parser.add_argument("model", type=str, help="Path to model file.")
    parser.add_argument("--headless", action="store_true", help="Run without displaying video.")
    args = parser.parse_args()

    model = TsModel().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    video = cv2.VideoCapture(args.video)
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

        if not args.headless:
            draw_frame(frame, cls)
            cv2.imshow("frame", frame)
            cv2.waitKey(int(1000 / fps))

        frame_index += 1

    pbar.close()

    with open("active_sections.txt", "w") as f:
        for start, end in active_sections:
            f.write(f"{start} {end}\n")


def vis_timestamps():
    """
    Alternative entry point:
    Draw multiple timestamps simultaneously, coloring in where each active section is.
    Makes it easy to compare ground truth vs pred.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("timestamps", nargs="+", help="Paths to timestamp files.")
    args = parser.parse_args()

    img = np.zeros((50 * len(args.timestamps), 1000, 3), dtype=np.uint8)
    img[:] = 255

    all_ts = []
    max_frame = 0
    for path in args.timestamps:
        ts = read_ts(path)
        all_ts.append(ts)
        max_frame = max(max_frame, max(x[1] for x in ts))

    for i, ts in enumerate(all_ts):
        for start, end in ts:
            start_px = int(start / max_frame * img.shape[1])
            end_px = int(end / max_frame * img.shape[1])
            cv2.rectangle(img, (start_px, i * 50), (end_px, (i + 1) * 50), (0, 255, 0), -1)

    cv2.imwrite("timestamps_vis.png", img)


if __name__ == "__main__":
    #main()
    vis_timestamps()
