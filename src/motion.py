"""
Farneback optical flow.
"""

import cv2
import numpy as np
import torch
from torchvision.utils import flow_to_image


class OpticalFlow:
    def __init__(self):
        self.prev_frame = None
        self.prev_flow = None

    def update(self, frame):
        """
        frame: cv2 format.
        return: ndarray float (H, W, 2)
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

        return self.prev_flow


def vis_of(frame, of):
    """
    frame: cv2 format.
    """
    cv2.imshow("Frame", frame)

    of = torch.from_numpy(of).permute(2, 0, 1)  # (2, H, W)
    vis = flow_to_image(of).permute(1, 2, 0).numpy()  # (H, W, 3)
    cv2.imshow("Optical Flow", vis)

    cv2.waitKey(1)
