"""
Perspective estimation module.
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import RANSACRegressor, LinearRegression
from transformers import pipeline

from ..utils.constants import *


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

        self.depth_nn = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

        self.ransac = RANSACRegressor(
            LinearRegression(),
            min_samples=30,
            max_trials=100,
            residual_threshold=10,
        )

    def update(self, frame, boxes, iter_i):
        """
        frame: cv2 format.
        boxes: boxes format.
        iter_i: CV pipeline iteration number.
        return: (person_locs, mask_locs)
            person_locs: ndarray float (N, 2) xy, physical locations of each box.
            mask_locs: ndarray float (M, 2) xy, field mask vertex locations.
        """
        run_nn = False
        if PERSP_INTERVAL == -1 and iter_i == 0:
            run_nn = True
        elif PERSP_INTERVAL > 0 and iter_i % PERSP_INTERVAL == 0:
            run_nn = True
        if run_nn:
            self.compute_vanishing(frame)

        # Pixel pos of boxes bottom edge.
        px_pos = np.stack((
            (boxes[:, 0] + boxes[:, 2]) / 2,
            boxes[:, 3],
        ), axis=1, dtype=float)
        person_locs = self.compute_locations(px_pos)

        mask_locs = self.compute_locations(self.mask_points)
        return person_locs, mask_locs

    def compute_vanishing(self, frame):
        """
        Computes:
        - self.vanishing: Y pixel pos of vanishing line (horizon).
        - self.dist_min: Distance from camera to ground location
            corresponding to the bottom of the frame.

        Run depth NN, linear regression, and trig/geometry formulas.
        """
        # Run "depth anything" NN.
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_map = self.depth_nn(frame)["predicted_depth"]

        # Fit linear model of depth vs Y pixel pos.
        depth_values = []
        y2s = []
        for y in range(int(depth_map.shape[0] * (1 - DEPTH_YLIMIT)),
                       depth_map.shape[0],
                       DEPTH_SAMPLING):
            depth_samps = depth_map[y, ::DEPTH_SAMPLING]
            depth_values.extend(depth_samps)
            y2s.extend(y for _ in range(len(depth_samps)))
        depth_values = np.array(depth_values)
        y2s = np.array(y2s)[:, None]

        # Linear regression.
        # height = m * y2_pos + b
        self.ransac.fit(y2s, depth_values)
        m = self.ransac.estimator_.coef_[0]
        b = self.ransac.estimator_.intercept_
        #vis_vanishing(y2s, depth_values, m, b, self.ransac.inlier_mask_)

        # Find vanishing, by setting height = 0
        self.vanishing = -b / m
        # Find min dist with trig.
        theta_bottom = np.pi / 2 - (1 - self.vanishing / DET_RES[1]) * self.vert_fov
        self.dist_min = CAM_HEIGHT / np.cos(theta_bottom)

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
