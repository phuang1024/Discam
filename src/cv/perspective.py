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

    def __init__(self, mask_path):
        self.mask_points = np.load(mask_path)
        self.mask_points[:, 0] *= DET_RES[0]
        self.mask_points[:, 1] *= DET_RES[1]

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

        px_x = (boxes[:, 0] + boxes[:, 2]) / 2
        px_y = boxes[:, 3]
        px_pos = np.stack((px_x, px_y), axis=1, dtype=float)
        locs = self.compute_locations(px_pos, True)
        mask_locs = self.compute_locations(self.mask_points, False)
        print(self.mask_points, mask_locs)
        vis_locations(locs, mask_locs)
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

    def compute_locations(self, px_pos, filter):
        """
        Compute XY physical locations of detections.
        px_pos: ndarray float (N, 2) xy, pixel positions.
        return: ndarray float (N, 2) xy, physical positions.
        """
        heights = self.m * px_pos[:, 1] + self.b
        hori_pos = (px_pos[:, 0] / DET_RES[0]) - 0.5

        dists = self.dist_min * self.height_max / heights
        y_pos = np.sqrt(np.pow(dists, 2) - CAM_HEIGHT ** 2)
        x_pos = hori_pos * 2 * dists * np.tan(self.hori_fov / 2)

        if filter:
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


def vis_locations(locs, mask_locs):
    def interp_coords(coords):
        coords[:, 0] = np.interp(coords[:, 0], (-40, 40), (0, 800))
        coords[:, 1] = np.interp(coords[:, 1], (0, 80), (800, 0))

    img = np.full((800, 800, 3), 255, dtype=np.uint8)

    interp_coords(mask_locs)
    mask_locs = mask_locs.astype(int)
    print(mask_locs)
    cv2.polylines(img, [mask_locs], True, (0, 0, 255), 4)

    interp_coords(locs)
    locs = locs.astype(int)
    for x, y in locs:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

    cv2.imshow("Locations", img)
