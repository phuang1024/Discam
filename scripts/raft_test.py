import argparse

import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image

import cv2

RES = (640, 360)

model = raft_small(weights=Raft_Small_Weights.DEFAULT).eval()


def resize_mul8(frame):
    width = frame.shape[1]
    height = frame.shape[0]
    new_w = (width // 8) * 8
    new_h = (height // 8) * 8
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


parser = argparse.ArgumentParser()
parser.add_argument("video")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)

last_frame = None

while True:
    for _ in range(8):
        ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, RES, interpolation=cv2.INTER_AREA)
    frame = resize_mul8(frame)

    # [C, H, W] [-1, 1]
    frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
    frame_t = frame_t.unsqueeze(0)

    if last_frame is not None:
        flow = model(last_frame, frame_t)[-1].squeeze(0)
        flow_vis = flow_to_image(flow)
        flow_vis = flow_vis.permute(1, 2, 0).numpy()
        cv2.imshow("flow", flow_vis)

    last_frame = frame_t

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
