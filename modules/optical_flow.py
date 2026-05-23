"""
Farneback dense optical flow.
"""

import argparse

import cv2
import numpy as np


class OpticalFlow:
    """
    Wrapper around optical flow.
    Keeps track of prev frame.
    """

    def __init__(self):
        self.prev_frame = None

    def compute(self, frame):
        """
        Compute optical flow between prev_frame and frame.
        If this is the first call, returns None.

        return: [H, W, 2] float32 tensor
            Last dimension is (dx, dy).
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = frame
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame,
            frame,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        self.prev_frame = frame
        return flow


def vis_flow_hsv(flow):
    """
    Visualize optical flow.
    Hue: direction. Value: magnitude.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hue = ang * 180 / np.pi / 2
    sat = np.ones_like(mag) * 255
    val = cv2.normalize(np.sqrt(mag), None, 0, 255, cv2.NORM_MINMAX)
    hsv = np.stack((hue, sat, val), axis=-1)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    from video_read import ScaledReader

    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    cap = ScaledReader(args.video)
    flow_comp = OpticalFlow()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flow = flow_comp.compute(frame)
        if flow is not None:
            vis = vis_flow_hsv(flow)
            cv2.imshow("frame", frame)
            cv2.imshow("flow", vis)

        if cv2.waitKey(10) == ord("q"):
            break
