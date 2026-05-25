"""
Motion analysis with optical flow and background removal.
"""

import cv2
import numpy as np
import torch

from utils import *


class Motion:
    def __init__(self):
        # Set every iteration in update.
        self.prev_frame = None
        self.prev_flow = None

        self.bg_remover = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=8,
            detectShadows=False,
        )

        self.of_ema = BlurFilter()
        self.bgr_ema = BlurFilter()

    def update(self, frame):
        """
        Run OF and BGR.
        frame: cv2 format.
        return: {
            of: Optical flow magnitude, ndarray float (H, W)
            bgr: Foreground mask, ndarray bool (H, W)
        }
        """
        # Optical flow.
        if self.prev_frame is None:
            self.prev_flow = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)
        else:
            self.prev_flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                self.prev_flow,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
        self.prev_frame = frame

        # Take magnitude.
        of = np.sqrt(self.prev_flow[:, :, 0] ** 2 + self.prev_flow[:, :, 1] ** 2)
        # Convert to binary mask.
        of = of > 0
        of = self.of_ema.update(of)

        # BG remove.
        fg_mask = self.bg_remover.apply(frame)
        fg_mask = fg_mask > 0
        fg_mask = self.bgr_ema.update(fg_mask)

        return {
            "of": self.prev_flow,
            "bgr": fg_mask,
        }


class BlurFilter:
    """
    """

    def __init__(self):
        # EMA of input.
        self.output = None

    def update(self, x):
        """
        x: ndarray bool (H, W)
        """
        x = x.astype(np.float32)
        for _ in range(MOTION_BLUR_PASSES):
            x = cv2.GaussianBlur(x, (MOTION_BLUR_SIZE, MOTION_BLUR_SIZE), 0)

        if self.output is None:
            self.output = x
        else:
            self.output = self.output * (1 - MOTION_EMA) + x * MOTION_EMA

        y = self.output > MOTION_THRES
        return y


def vis_motion(frame, motion_out):
    """
    frame: cv2 format.
    motion_out: output of Motion.update
    """
    frame = frame.copy()

    overlay = np.zeros_like(frame)
    # BG remover.
    overlay[motion_out["bgr"]] = (255, 255, 255)
    # Optical flow.
    dy = motion_out["of"][:, :, 0]
    dx = motion_out["of"][:, :, 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    #overlay[mag > 0, 1] = (255, 255, 255)

    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    cv2.imshow("Motion", frame)
    cv2.waitKey(1)
