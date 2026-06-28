"""
Perspective estimation module.
"""

import cv2
import numpy as np

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
        self.mask_points = np.load(mask_path)
        self.mask_points[:, 0] *= DET_RES[0]
        self.mask_points[:, 1] *= DET_RES[1]

        # FOV in rad.
        self.hori_fov = np.radians(CAM_FOV)
        self.vert_fov = self.hori_fov / DET_RES[0] * DET_RES[1]

        # Queue of detected boxes per frame.
        self.data = []

    def update(self, boxes):
        """
        boxes: tracked boxes format.
        return: m parameter.
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
        locs = self.compute_locations(px_pos)

        if True:
            mask_locs = self.compute_locations(self.mask_points)
            vis_locations(locs, mask_locs)

        return locs

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
        # height = m * y2_pos + b
        m, b = np.polyfit(y2s, heights, 1)

        # Find vanishing, by setting height = 0
        self.vanishing = -b / m
        # Find min dist with trig.
        theta_bottom = np.pi / 2 - (1 - self.vanishing / DET_RES[1]) * self.vert_fov
        self.dist_min = CAM_HEIGHT / np.cos(theta_bottom)

        #vis_vanishing(y2s, heights, m, b)

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


def vis_vanishing(xs, ys, m, b):
    """
    Visualize linear regression to compute vanishing point.
    """
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
    """
    Visualize computed locations.
    """
    RES = 800

    def interp_coords(coords):
        """From XY location to vis image pixel pos."""
        return (
            np.interp(coords[:, 0], (-40, 40), (0, RES)),
            np.interp(coords[:, 1], (0, 80), (RES, 0)),
        )

    img = np.full((RES, RES, 3), 255, dtype=np.uint8)

    mask_locs = interp_coords(mask_locs)
    mask_locs = np.array(mask_locs, dtype=int).swapaxes(0, 1)
    cv2.polylines(img, [mask_locs], True, (0, 0, 255), 4)

    xs, ys = interp_coords(locs)
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)

    cv2.imshow("Locations", img)
    cv2.waitKey(1)
