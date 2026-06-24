"""
Active player classification module.
"""

import cv2
import numpy as np

from utils.constants import *
from utils.field_mask import create_mask


class Classifier:
    """
    Classification with manual field mask and threshold.
    """

    def __init__(self, mask_path):
        # Load and blur field mask. Convert to [-1, 1] range.
        points = np.load(mask_path)
        self.field_mask = create_mask(points, DET_RES)
        self.field_mask = cv2.blur(self.field_mask, (50, 50))
        self.field_mask = self.field_mask / 127.5 - 1

    def update(self, detector_out):
        """
        detector_out: Detector output for current frame.
        return: List of indices of active player boxes of detector_out.
        """
        indices = []
        for i, box in enumerate(detector_out):
            x1, y1, x2, y2 = box
            mid_x = (x1 + x2) // 2
            if self.field_mask[y2, mid_x] > 0:
                indices.append(i)
        return indices
