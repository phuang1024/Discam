"""
Perspective estimation module.
TODO this should be made streamable!
"""

import numpy as np

from utils import *


def compute_vanishing(detector_out):
    """
    Fit linear model to box height by Y position
    using detector outputs.
    detector_out: List of detector outputs.
    return: m, b, y0, height_max
        m, b: height = m * y + b
            In pixels, wrt image coordinates, with y=0 at the top.
        y0: Y position where box height becomes 0.
        height_max: Box height at the bottom of the image.
    """
    # Gather data. X is "box y2 position". Y is "box height".
    xs = []
    ys = []
    for data in detector_out:
        for box in data["boxes"]:
            xs.append(box[3])
            ys.append(box[3] - box[1])
    # h = m * y + b
    m, b = np.polyfit(xs, ys, 1)

    # Extract output params.
    y0 = -b / m
    height_max = m * NN_RES[1] + b
    vis_vanishing(xs, ys, m, b)
    return m, b, y0, height_max


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
