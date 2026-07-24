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

    def __init__(self, live_mode):
        self.live_mode = live_mode

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
        # Run initial field mask classify.
        # These are binary masks, same len as ``person_locs``.
        def_pos, maybe_pos = self.mask_classify(person_locs, mask_locs)

        # Update KNN.
        self.update_knn(person_locs[def_pos])

        # Run Z score filter.
        # If people are close together, can filter outliers from within "definitely" positive.
        do_def_filter = self.sep_metric < FILTER_DEF_THRES
        active_inds = stddev_filter(person_locs, def_pos, maybe_pos, do_def_filter)
        return active_inds

    def mask_classify(self, person_locs, mask_locs):
        """Classify by proximity to field mask.

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


def stddev_filter(person_locs, def_pos, maybe_pos, filter_def):
    """Filter active players by std dev.

    First, optionally filter within ``def_pos``:
    People too far from mean are set to negative.

    Then, filter using ``maybe_pos``:
    People close enough are set to positive.

    Args:
        filter_def: Whether to filter within definitely positive.

    Returns:
        ``ndarray bool (N,)``, same length as ``person_locs``,
        final classification of whether someone should be tracked.
    """
    # Filter within definitely pos.
    if filter_def:
        mean, std = compute_dist(person_locs[def_pos])
        zs = compute_zs(person_locs, mean, std)
        def_pos = np.logical_and(def_pos, zs < ACTIVE_STD_THRES)

    # Filter maybe pos by z score. Add close ones to positive.
    mean, std = compute_dist(person_locs[def_pos])
    zs = compute_zs(person_locs, mean, std)
    ret = np.logical_or(
        def_pos,
        np.logical_and(maybe_pos, zs < ACTIVE_STD_THRES),
    )
    return ret


def compute_dist(locs):
    """Compute mean and std of a set of 2D points.

    Args:
        locs: ``ndarray float (N, 2)``.

    Returns:
        ``(mean, std)``. Both are tuples length 2 xy.
    """
    xs = locs[:, 0]
    ys = locs[:, 1]
    return (
        (np.mean(xs), np.mean(ys)),
        (np.std(xs), np.std(ys)),
    )


def compute_zs(locs, mean, std):
    """Compute Z scores of ``locs`` wrt ``mean`` and ``std``.
    Args same format as ``compute_dist``.

    Return:
        ``ndarray float (N,)``, list of scalar Z scores.
    """
    zs = (locs - mean) / std
    zs = np.linalg.norm(zs, axis=1)
    return zs
