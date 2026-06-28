"""
Perspective estimation module.
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

from utils.constants import *


class ComputePersp:
    """
    Camera perspective estimation.
    Estimate vanishing point with linear model of box heights.
    Project pixel positions to locations on the field with model and other constants.

    "Location" refers to physical location on the field.
    "Pixel position" refers to position on image frame.
    """

    def __init__(self, mask_path):
        # For visualization.
        self.mask_points = np.load(mask_path)
        self.mask_points[:, 0] *= DET_RES[0]
        self.mask_points[:, 1] *= DET_RES[1]

        # FOV in rad.
        self.hori_fov = np.radians(CAM_FOV)
        self.vert_fov = self.hori_fov / DET_RES[0] * DET_RES[1]

        # Queue of detected boxes per frame.
        self.data = []
        # Initial model.
        self.vanishing = None
        self.dist_min = None

        self.ransac = RANSACRegressor(
            LinearRegression(),
            min_samples=30,
            max_trials=100,
            residual_threshold=10,
        )

    def update(self, boxes):
        """
        boxes: tracked boxes format.
        return: (player_locs, mask_locs)
            player_locs: ndarray float (N, 2) xy, physical locations of each box.
            mask_locs: ndarray float (M, 2) xy, field mask vertex locations.
        """
        # Add data to queue.
        self.data.append(boxes)
        if len(self.data) > PERSP_QSIZE:
            self.data.pop(0)

        self.compute_vanishing()

        # Pixel pos of boxes bottom edge.
        px_pos = np.stack((
            (boxes[:, 0] + boxes[:, 2]) / 2,
            boxes[:, 3],
        ), axis=1, dtype=float)
        player_locs = self.compute_locations(px_pos)

        mask_locs = self.compute_locations(self.mask_points)

        return player_locs, mask_locs

    def compute_vanishing(self):
        """
        Uses self.data
        Computes:
        - vanishing: Y pixel pos of vanishing line (horizon).
        - dist_min: Distance from camera to ground location
            corresponding to the bottom of the frame.
        """
        # Fit linear model of box height vs Y pixel pos.
        heights = []
        y2s = []
        for box_list in self.data:
            for box in box_list:
                heights.append(box[3] - box[1])
                y2s.append(box[3])
        heights = np.array(heights)
        y2s = np.array(y2s)[:, None]

        # height = m * y2_pos + b
        self.ransac.fit(y2s, heights)
        m = self.ransac.estimator_.coef_[0]
        b = self.ransac.estimator_.intercept_
        #vis_vanishing(y2s, heights, m, b, self.ransac.inlier_mask_)

        # Find vanishing, by setting height = 0
        vanishing = -b / m
        # Find min dist with trig.
        theta_bottom = np.pi / 2 - (1 - vanishing / DET_RES[1]) * self.vert_fov
        dist_min = CAM_HEIGHT / np.cos(theta_bottom)

        if self.vanishing is None:
            self.vanishing = vanishing
            self.dist_min = dist_min
        else:
            self.vanishing = PERSP_EMA * vanishing + (1 - PERSP_EMA) * self.vanishing
            self.dist_min = PERSP_EMA * dist_min + (1 - PERSP_EMA) * self.dist_min

    def compute_locations(self, px_pos):
        """
        Convert pixel pos to locations.
        px_pos: ndarray float (N, 2) xy, pixel positions.
        return: ndarray float (N, 2) xy, physical locations.
        """
        # Distance from cam, using linearity of size up to vanishing point.
        size_facs = np.interp(px_pos[:, 1], (self.vanishing, DET_RES[1]), (0, 1))
        size_facs = np.clip(size_facs, PERSP_MIN_SIZE, 1)
        dists = self.dist_min / size_facs

        # Y loc, using Pythag.
        y_locs = np.sqrt(np.pow(dists, 2) - CAM_HEIGHT ** 2)

        # X loc, using trig.
        # Horizontal pixel position normalized [-0.5, 0.5]
        hori_pos = px_pos[:, 0] / DET_RES[0] - 0.5
        x_locs = hori_pos * 2 * dists * np.tan(self.hori_fov / 2)

        ret = np.stack((x_locs, y_locs), axis=1)
        return ret


def vis_vanishing(xs, ys, m, b, inliers):
    """
    Visualize linear regression to compute vanishing point.
    """
    import matplotlib.pyplot as plt
    # Plot data
    plt.scatter(xs[inliers], ys[inliers], label="Data", alpha=0.5, color="blue")
    plt.scatter(xs[np.logical_not(inliers)], ys[np.logical_not(inliers)], label="Data", alpha=0.5, color="red")

    # Plot fitted line
    x_min = min(xs) - 100
    x_max = max(xs) + 100
    x_fit = np.array([x_min, x_max])
    y_fit = m * x_fit + b
    plt.plot(x_fit, y_fit, color="red", label="Fitted line")
    plt.xlabel("Y position")
    plt.ylabel("Height")
    plt.show()
