"""
Perspective estimation module.
"""

import cv2
import numpy as np

from utils.constants import *


class ComputePersp:
    """
    TODO
    Fit linear model to box height by Y position using detected boxes.
    Updates every N detector frames.

    Vanishing point model format:
        height = m * y2_pos + b
        height: Box height at some y position.
        y2: Pixel position of bottom box edge.
        m, b: Model params.
    """

    def __init__(self):
        # FOV in rad.
        self.hori_fov = np.radians(CAM_FOV)
        self.vert_fov = self.hori_fov / DET_RES[0] * DET_RES[1]

        # Queue of detected boxes per frame.
        self.data = []

        # Vanishing point model params (see above).
        self.m = 0
        self.b = 0

    def update(self, boxes):
        """
        boxes: tracked boxes format.
        return: m parameter.
        """
        self.data.append(boxes)
        if len(self.data) > PERSP_QSIZE:
            self.data.pop(0)

        self.compute_vanishing()
        locs = self.compute_locations(boxes)
        vis_locations(locs)
        return locs

    def compute_vanishing(self):
        """
        Uses self.data
        """
        heights = []
        y2s = []
        for box_list in self.data:
            for box in box_list:
                heights.append(box[3] - box[1])
                y2s.append(box[3])
        # height = m * y2_pos + b
        m, b = np.polyfit(y2s, heights, 1)
        self.m = m
        self.b = b

        self.height_max = m * DET_RES[1] + b
        self.vanishing = -b / m
        theta_bottom = np.pi / 2 - (1 - self.vanishing / DET_RES[1]) * self.vert_fov
        self.dist_min = CAM_HEIGHT / np.cos(theta_bottom)

        #vis_vanishing(y2s, heights, self.m, self.b)

    def compute_locations(self, boxes):
        """
        Compute XY physical locations of detections.
        boxes: tracked boxes format.
        return: ndarray float (N, 2) xy
        """
        boxes = boxes.astype(float)
        heights = self.m * boxes[:, 3] + self.b
        hori_pos = (boxes[:, 2] + boxes[:, 0]) / 2
        hori_pos = (hori_pos / DET_RES[0]) - 0.5

        dists = self.dist_min * self.height_max / heights
        y_pos = np.sqrt(np.pow(dists, 2) - CAM_HEIGHT ** 2)
        x_pos = hori_pos * 2 * dists * np.tan(self.hori_fov / 2)

        good_inds = heights > self.height_max / 10
        x_pos = x_pos[good_inds]
        y_pos = y_pos[good_inds]

        ret = np.stack((x_pos, y_pos), axis=1)
        return ret


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


def vis_locations(locs):
    img = np.full((800, 800, 3), 255, dtype=np.uint8)
    for x, y in locs:
        px_x = int(np.interp(x, (-40, 40), (0, 800)))
        px_y = int(np.interp(y, (0, 80), (800, 0)))
        cv2.circle(img, (px_x, px_y), 3, (255, 0, 0), -1)
    cv2.imshow("Locations", img)
