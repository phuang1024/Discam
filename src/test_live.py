"""
Stream a video, and display inference results in real time.
"""

import argparse

import cv2
import numpy as np
import torch

from constants import *
from model import DiscamModel


def make_vis_image(pred):
    """
    Makes image visualizing prediction.
    """
    img = np.full((400, 400, 3), 255, dtype=np.uint8)

    # Draw text labels
    cv2.putText(img, f"tx: {pred[0]:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, f"ty: {pred[1]:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, f"scale: {pred[2]:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(img, (10, 150), (150, 390), (0, 0, 0), 1)
    x = int(pred[0] * 70 + 80)
    y = int(pred[1] * 120 + 270)
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    y = int(pred[2] * 120 + 270)
    cv2.circle(img, (360, y), 5, (0, 255, 0), -1)

    return img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=int, default=1)
    args = parser.parse_args()

    model = DiscamModel().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    video = cv2.VideoCapture(args.video)

    while True:
        for _ in range(args.interval):
            ret, frame = video.read()
            if not ret:
                return

        x = torch.tensor(frame[..., ::-1] / 255, device=DEVICE, dtype=torch.float32)
        x = x.permute(2, 0, 1).unsqueeze(0)
        y = model(x).squeeze().cpu().numpy()

        vis = make_vis_image(y)
        cv2.imshow("frame", frame)
        cv2.imshow("vis", vis)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
