"""
Active player classification module.
"""

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from ..utils.constants import *


class Classifier:
    """
    Classify whether each player is active.
    Uses manual field mask, computed physical locations, and data analysis.
    """

    def __init__(self):
        # KNN to detect the pull (Frisbee), when two teams are far apart.
        self.knn = KMeans(2)

    def update(self, person_locs, mask_locs):
        """
        person_locs: ndarray float (N, 2) xy
        mask_locs: ndarray float (M, 2) xy
        return: ndarray bool, same length as `person_locs`.
            Whether each box is active.
        """
        # Run initial field mask filter.
        person_locs = person_locs.astype(np.float32)
        mask_locs = mask_locs.astype(np.float32)
        active_inds, do_filter = self.filter_field_mask(person_locs, mask_locs)

        active_locs = person_locs[active_inds]
        self.update_knn(active_locs)

        active_inds = stddev_filter(person_locs, active_inds, do_filter)
        return active_inds

    def update_knn(self, active_locs):
        """
        Update KNN and separation metric.
        """
        def variance(data):
            return np.sqrt(np.sum(np.var(data, axis=0)))

        self.knn.fit(active_locs)

        global_std = variance(active_locs)
        cls1_std = variance(active_locs[self.knn.labels_ == 0])
        cls2_std = variance(active_locs[self.knn.labels_ == 1])

        self.sep_metric = global_std / (cls1_std + cls2_std)

    def filter_field_mask(self, person_locs, mask_locs):
        """
        Filter by proximity to field mask.
        person_locs: Physical locations of all detected people.
        mask_locs: Physical locations of mask points.
        return: (active_inds, do_filter)
            Categorizations by 2 thresholds. See constants and docs.
        """
        active_inds = np.zeros([len(person_locs)], dtype=bool)
        do_filter = np.zeros_like(active_inds)
        for i, loc in enumerate(person_locs):
            dist = cv2.pointPolygonTest(mask_locs, loc, True)
            if dist > POS_THRES:
                # Definitely active.
                active_inds[i] = True
            elif dist > MAYBE_POS_THRES:
                # Maybe active.
                do_filter[i] = True

        return active_inds, do_filter


def stddev_filter(person_locs, active_inds, do_filter):
    """
    Filtering with mean and stddev of given active players.
    """
    xs = person_locs[active_inds, 0]
    ys = person_locs[active_inds, 1]
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    x_std = np.std(xs)
    y_std = np.std(ys)

    ret = active_inds.copy()
    for i, (x, y) in enumerate(person_locs):
        if do_filter[i]:
            zx = (x - x_mean) / x_std
            zy = (y - y_mean) / y_std
            z_score = np.hypot(zx, zy)
            ret[i] = z_score < ACTIVE_STD_THRES
    return ret
