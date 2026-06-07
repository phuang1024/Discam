"""
Motion analysis with optical flow and background removal.
"""

import cv2
import numpy as np
import torch
from torchvision.utils import flow_to_image

from field_mask import read_mask
from utils import *


class Motion:
    def __init__(self, field_mask_path):
        # Set every iteration in update.
        self.prev_frame = None
        # This is raw output of Farneback.
        self.prev_flow = np.zeros((RES[1], RES[0], 2), dtype=np.float32)

        self.of_median_filter = TemporalMedianFilter()
        self.of_salience = VelocitySalience()
        self.persp_scale_img = self.make_persp_scale(field_mask_path)

    def update(self, frame):
        """
        frame: cv2 format.
        return: {
        }
        """
        # Compute optical flow.
        if self.prev_frame is not None:
            self.prev_flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                self.prev_flow,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
        self.prev_frame = frame

        of = self.of_median_filter.update(self.prev_flow)
        of = of * self.persp_scale_img
        #of_salience = self.of_salience.update(of)

        return {
            "of": of,
            #"salience": of_salience,
        }

    def make_persp_scale(self, field_mask_path):
        """
        Make a multiplicative per-pixel scaling for OF,
        to correct for far people being smaller.
        return: ndarray float (H, W, 1)
        """
        if field_mask_path is None:
            return np.ones((RES[1], RES[0], 1), dtype=np.float32)

        mask = read_mask(field_mask_path)
        min_y = np.min(mask[:, 1]) * RES[1]

        scale_img = np.zeros((RES[1], RES[0]), dtype=np.float32)
        for y in range(RES[1]):
            scale_img[y] = interp(y, min_y, RES[1], OF_PERSP_SCALE, 1, clamp=True)

        return scale_img[..., None]


class TemporalMedianFilter:
    """
    Applied on raw OF output, to smooth noise from camera jitters.
    """

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.frames = np.empty([window_size, RES[1], RES[0], 2], dtype=np.float32)
        self.index = 0

    def update(self, frame):
        self.frames[self.index] = frame
        self.index = (self.index + 1) % self.window_size

        median_frame = np.median(self.frames, axis=0)
        return median_frame


class VelocitySalience:
    """
    Keep a salience map (likelihood of being active)
    based on velocity magnitude.
    vel_mag = norm(of)
    accel_mag = norm(of - EMA(of))
    """

    def __init__(self):
        self.salience = np.zeros((RES[1], RES[0]), dtype=np.float32)
        # EMA of OF output.
        self.accel_ema = EMA(0.2)

    def update(self, of):
        vel_mag = np.linalg.norm(of, axis=-1)

        self.accel_ema.update(of)
        accel_ema = cv2.dilate(self.accel_ema.value, np.ones((3, 3), dtype=np.float32), iterations=3)
        accel_mag = np.linalg.norm(of - accel_ema, axis=-1)

        curr_salience = vel_mag + 2 * accel_mag
        self.salience = np.maximum(self.salience, curr_salience)
        self.salience *= 0.9
        return self.salience


def vis_motion(frame, motion_out):
    """
    frame: cv2 format.
    motion_out: output of Motion.update
    """

    cv2.imshow("Frame", frame)

    of = motion_out["of"]
    vis = torch.from_numpy(of).permute(2, 0, 1)  # (2, H, W)
    vis = flow_to_image(vis).permute(1, 2, 0).numpy()  # (H, W, 3)
    cv2.imshow("OF", vis)

    """
    vis = motion_out["salience"]
    vis = np.clip(vis / 10, 0, 1)
    cv2.imshow("salience", vis)
    """

    cv2.waitKey(1)


def vis_of_magnitude(of):
    """
    Visualize magnitude floor.
    of: ndarray float (H, W, 2)
    """
    mag = np.linalg.norm(of, axis=-1)
    max_mag = np.max(mag)

    while True:
        for floor in np.linspace(0, max_mag, 10):
            print("Floor", floor)
            mask = mag > floor

            curr_of = of * mask[..., None]
            curr_of = torch.from_numpy(curr_of).permute(2, 0, 1)
            vis = flow_to_image(curr_of).permute(1, 2, 0).numpy()
            cv2.imshow("OF magnitude floor", vis)
            cv2.waitKey(1000)
