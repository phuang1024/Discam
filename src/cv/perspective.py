"""
Perspective estimation module.
TODO this should be made streamable!
"""

import numpy as np

from utils.constants import *


class ComputePersp:
    """
    Fit linear model to box height by Y position using detected boxes.
    Updates every N detector frames.

    Model format:
        height_scale = -m * (RES[1] - y2)
        height_scale <= 1: 1 at the bottom of the screen, decreases linearly moving up.
            Height of boxes at y as fraction of box height at bottom.
        y2 in (0, RES[1]): Pixel position of bottom box edge.
        m > 0: scale per px
    """

    def __init__(self):
        # Append to this each frame. When it reaches set interval, compute new model.
        self.data = []

        # Default model: Assume 1/3x scale halfway up image.
        self.m = 1/3 / (DET_RES[1] / 2)

    def update(self, boxes):
        """
        boxes: tracked boxes format.
        return: m parameter.
        """

    def compute_param(self):
        """
        Uses self.data
        """
        heights = []
        y2s = []
        for box_list in self.data:
            for box in box_list:
                heights.append(box[3] - box[1])
                y2s.append(box[3])
        # height = a * y2_pos + b
        a, b = np.polyfit(y2s, heights, 1)

        # Extract output param.
        height_max = a * DET_RES[1] + b
        m = -a / height_max
        return m


def vis_vanishing(xs, ys, m, b):
    import matplotlib.pyplot as plt
    # Plot data
    plt.scatter(xs, ys, label="Data", alpha=0.5)
    # Plot fitted line
    x_min = min(xs) - 100
    x_max = max(xs) + 100
    x_fit = np.array([x_min, x_max])
    y_fit = m * x_fit + b
    plt.plot(x_fit, y_fit, color="red", label="Fitted line")
    plt.xlabel("Y position")
    plt.ylabel("Height")
    plt.show()
