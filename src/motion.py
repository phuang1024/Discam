"""
Motion analysis with optical flow and background removal.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import flow_to_image

from utils import *


class Motion:
    def __init__(self):
        # Set every iteration in update.
        self.prev_frame = None
        self.prev_flow = None

        """
        self.bg_remover = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=8,
            detectShadows=False,
        )
        """

        #self.of_filter = OpticalFlowFilter()

    def update(self, frame):
        """
        Run OF and BGR.
        frame: cv2 format.
        return: {
            of: Optical flow mask, ndarray float (H, W, 2)
            of_salience: Output of OpticalFlowFilter. ndarray float (H, W) [0, 1]
            bgr: Foreground mask, ndarray bool (H, W)
        }
        """
        # Compute optical flow.
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

        #of_salience = self.of_filter.update(self.prev_flow)

        # BG remove.
        """
        fg_mask = self.bg_remover.apply(frame)
        fg_mask = fg_mask > 0
        fg_mask = self.bgr_ema.update(fg_mask)
        """

        return {
            "of": self.prev_flow,
            #"of_salience": of_salience,
            #"bgr": fg_mask,
        }


class OpticalFlowFilter:
    """
    Filter to detect fast moving people.

    1. Preprocessing: Magnitude multiplication to account for apparent size,
        and blur.
    TODO some filtering for bg movement
    2. Analysis: We keep a patch grid of salience. Patches, to reduce computational load.
        Fast moving objects increase salience.
        All moving objects move existing, corresponding salience.
    """

    def __init__(self):
        self.salience = None

    def update(self, of):
        """
        of: ndarray float (H, W, 2)
        """
        patch_w = of.shape[1] // OF_PATCH_SIZE
        patch_h = of.shape[0] // OF_PATCH_SIZE
        of = cv2.resize(of, (patch_w, patch_h))

        # Create empty on first iter.
        if self.salience is None:
            self.salience = np.zeros([patch_h, patch_w], dtype=float)

        # Decay salience.
        self.salience *= 1 - OF_DECAY_FAC

        magnitude = np.sqrt(of[..., 0] ** 2 + of[..., 1] ** 2)

        # Find fast areas. Curve using tanh.
        fast = magnitude * (magnitude > OF_FAST_THRES)
        fast = np.tanh((fast - OF_FAST_THRES) * OF_FAST_SCALE)
        self.salience = np.maximum(self.salience, fast)

        # Apply of.
        #self.salience = self.apply_flow(of, self.salience)

        return self.salience


def vis_motion(frame, motion_out):
    """
    frame: cv2 format.
    motion_out: output of Motion.update
    """
    # of
    of = motion_out["of"]
    of = torch.from_numpy(of).permute(2, 0, 1)  # (2, H, W)
    vis = flow_to_image(of).permute(1, 2, 0).numpy()  # (H, W, 3)
    cv2.imshow("Optical Flow", vis)

    # Plot histogram of OF magnitude.
    """
    of = motion_out["of"]
    mag = np.sqrt(of[..., 0] ** 2 + of[..., 1] ** 2)
    mag = mag.flatten()
    plt.clf()
    plt.hist(mag, bins=100, range=(0, 10))
    # Log y axis
    plt.yscale("log")
    plt.pause(0.001)
    """

    # of_salience
    """
    img = motion_out["of_salience"]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, (vis.shape[1], vis.shape[0]), cv2.INTER_NEAREST)
    cv2.imshow("OF salience", img)
    """

    cv2.waitKey(1)
