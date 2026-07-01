"""Classification module.
"""

import cv2
import numpy as np
import scipy.spatial.distance
from sklearn.cluster import KMeans

from ..utils.constants import *


class Classifier:
    """Classify whether each player is active.
    Uses field mask, computed physical ``location``, and data analysis.
    Separation metric for Trim is also computed here.
    """

    def __init__(self):
        # KNN to detect the pull (Frisbee), when two teams are far apart.
        self.knn = KMeans(2)
        # Metric used in Trim. Computed every iteration.
        self.sep_metric = 0

    def update(self, person_locs, mask_locs):
        """
        Args:
            person_locs: ``ndarray float (N, 2)``, xy locations.
            mask_locs: ``ndarray float (M, 2)``, xy locations.

        Returns:
            ``ndarray bool (N,)``, whether each box is active.
        """
        # Run initial field mask filter.
        active_inds, do_filter = self.filter_field_mask(person_locs, mask_locs)

        # Update KNN.
        active_locs = person_locs[active_inds]
        self.update_knn(active_locs)

        # Run Z score filter.
        active_inds = stddev_filter(person_locs, active_inds, do_filter)
        return active_inds

    def update_knn(self, active_locs):
        """Update KNN and separation metric.
        """
        if len(active_locs) < 2:
            return
        self.knn.fit(active_locs)

        # Shape (N, 2).
        points1 = active_locs[self.knn.labels_ == 0]
        points2 = active_locs[self.knn.labels_ == 1]
        if len(points1) <= 2 or len(points2) <= 2:
            return

        # Find mean and covs of each group.
        mean1 = np.mean(points1, axis=0)
        mean2 = np.mean(points2, axis=0)
        eye = np.eye(2)
        cov1 = np.cov(points1.swapaxes(0, 1)) + eye * SEP_EPS
        cov2 = np.cov(points2.swapaxes(0, 1)) + eye * SEP_EPS

        # Find Z score of residual.
        resid = mean2 - mean1
        zeros = np.zeros_like(resid)
        try:
            zscore1 = scipy.spatial.distance.mahalanobis(resid, zeros, np.linalg.inv(cov1))
            zscore2 = scipy.spatial.distance.mahalanobis(-resid, zeros, np.linalg.inv(cov2))
        except np.linalg.LinAlgError:
            return
        self.sep_metric = (zscore1 + zscore2) / 2

    def filter_field_mask(self, person_locs, mask_locs):
        """Filter by proximity to field mask.

        Returns:
            ``(def_pos, maybe_pos)``. Both are ``ndarray bool (N,)``.
            Classifications by 2 thresholds. See constants and docs.
        """
        mask_locs = mask_locs.astype(int)
        def_pos = np.zeros([len(person_locs)], dtype=bool)
        maybe_pos = np.zeros_like(def_pos)
        for i, loc in enumerate(person_locs):
            dist = cv2.pointPolygonTest(mask_locs, loc, True)
            if dist > DEF_POS_THRES:
                def_pos[i] = True
            elif dist > MAYBE_POS_THRES:
                maybe_pos[i] = True
        return def_pos, maybe_pos


def stddev_filter(person_locs, def_pos, maybe_pos):
    """Filtering with mean and stddev of definitely active players.
    "Maybe" active players within some Z score of "definitely" are also marked active.
    """
    # Get "definitely" distribution.
    xs = person_locs[def_pos, 0]
    ys = person_locs[def_pos, 1]
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    x_std = np.std(xs)
    y_std = np.std(ys)

    ret = def_pos.copy()
    for i, (x, y) in enumerate(person_locs):
        if maybe_pos[i]:
            zx = (x - x_mean) / x_std
            zy = (y - y_mean) / y_std
            z_score = np.hypot(zx, zy)
            ret[i] = z_score < ACTIVE_STD_THRES
    return ret
