"""
Active player classification module.
"""

import cv2
import numpy as np

from ..utils.constants import *


class Classifier:
    """
    Classify whether each player is active.
    Uses manual field mask, computed physical locations, and data analysis.
    """

    def update(self, player_locs, mask_locs):
        """
        player_locs: ndarray float (N, 2) xy
        mask_locs: ndarray float (M, 2) xy
        return: ndarray bool, same length as `player_locs`.
            Whether each box is active.
        """
        player_locs = player_locs.astype(np.float32)
        mask_locs = mask_locs.astype(np.float32)

        active_inds = np.zeros([len(player_locs)], dtype=bool)
        do_filter = np.zeros_like(active_inds)
        for i, loc in enumerate(player_locs):
            dist = cv2.pointPolygonTest(mask_locs, loc, True)
            if dist > POS_THRES:
                # Definitely active.
                active_inds[i] = True
            elif dist > MAYBE_POS_THRES:
                # Maybe active.
                do_filter[i] = True

        active_inds = stddev_filter(player_locs, active_inds, do_filter)
        return active_inds


def stddev_filter(player_locs, active_inds, do_filter):
    """
    Filtering with mean and stddev of given active players.
    """
    xs = player_locs[active_inds, 0]
    ys = player_locs[active_inds, 1]
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    x_std = np.std(xs)
    y_std = np.std(ys)

    ret = active_inds.copy()
    for i, (x, y) in enumerate(player_locs):
        if do_filter[i]:
            zx = (x - x_mean) / x_std
            zy = (y - y_mean) / y_std
            z_score = np.hypot(zx, zy)
            ret[i] = z_score < ACTIVE_STD_THRES
    return ret
