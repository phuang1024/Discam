"""
Active player classification module.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from utils.constants import *
from utils.field_mask import create_mask, create_persp_scale


@dataclass
class Track:
    """
    Simple track dataclass.
    Add points through add_point()
        Keeps track of running average.
    """
    points: list
    """List of (x, y) box lower edge points."""
    last_update: int
    """Frame number last updated."""
    _vel_moment: float = 0

    def add_point(self, p):
        self.points.append(p)
        if len(self.points) >= 2:
            p1 = self.points[-1]
            p2 = self.points[-2]
            self._vel_moment += np.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_avg_vel(self):
        """
        average(norm(p2 - p1)) across all points.
        """
        if len(self.points) <= 1:
            return 0
        return self._vel_moment / (len(self.points) - 1)


class Classifier:
    """
    Classification with manual field mask and threshold.
    """

    def __init__(self, mask_path):
        # Load and blur field mask. Convert to [-1, 1] range.
        points = np.load(mask_path)
        self.field_mask = create_mask(points, DET_RES)
        self.field_mask = cv2.blur(self.field_mask, (50, 50))
        self.field_mask = self.field_mask / 127.5 - 1

        self.persp_scale = create_persp_scale(self.field_mask, DET_RES, PERSP_SCALE)

        #self.tracks = {}

    def update(self, boxes, frame_i):
        """
        boxes: tracked boxes format. From detector.
        return: ndarray bool, same length as `boxes`.
            Whether each box is active.
        """
        #self.update_tracks(boxes, frame_i)

        active_inds = np.zeros([len(boxes)], dtype=bool)
        do_filter = np.zeros_like(active_inds)
        for i, box in enumerate(boxes):
            # Query mask at lower edge of box.
            x_mid = (box[0] + box[2]) // 2
            scale_value = self.persp_scale[box[3], x_mid]
            mask_value = self.field_mask[box[3], x_mid] / scale_value
            active_inds[i] = mask_value > FIELD_MASK_THRES
            do_filter[i] = -0.5 < mask_value < 0.5

            # Check if velocity is too low.
            """
            if box[4] in self.tracks:
                vel = self.tracks[box[4]].get_avg_vel()
                #vel = (vel + 1) / len(self.tracks[box[4]].points)
                if len(self.tracks[box[4]].points) >= 4 and vel < 5:
                    active = False
            """

        active_inds = self.knn_median_filter(boxes, active_inds, do_filter, KNN_NUM)
        return active_inds

    def knn_median_filter(self, boxes, active_inds, do_filter, k):
        """
        Apply KNN median filter on boxes near boundary.
        For each such box, find k nearest boxes (including itself).
            Then use majority decision.
        X and Y distance are weighted differently, to account for perspective.
            Additionally, apply perspective scaling.

        boxes: tracked boxes format.
        active_inds: Box classification from field mask.
        do_filter: Binary array of whether to filter each box.
        k: Number of neighbors (including itself).
        """
        ret = active_inds.copy()
        for i, box1 in enumerate(boxes):
            if not do_filter[i]:
                continue

            # List of (box2_ind, dist)
            dists = []
            for j, box2 in enumerate(boxes):
                dists.append((j, self.compute_distance(box1, box2)))
            dists.sort(key=lambda x: x[1])

            active_count = 0
            for j in range(k):
                active_count += active_inds[dists[j][0]]
            ret[i] = active_count > k / 2

        return ret

    def compute_distance(self, box1, box2):
        """
        Compute distance between lower edge centerpoints.
        Applies both XY and Persp scaling.
        Persp scaling is based on box1 position.
        """
        b1x = (box1[0] + box1[2]) // 2
        b2x = (box2[0] + box2[2]) // 2
        dx = b1x - b2x
        dy = box1[3] - box2[3]
        dist = np.hypot(dx, dy * YX_SCALE)
        dist *= self.persp_scale[box1[3], b1x]
        return dist

    def update_tracks(self, boxes, frame_i):
        """
        Update self.tracks
        Removes stale tracks.
        boxes: tracked boxes format.
        """
        for box in boxes:
            id = box[4]
            if id == -1:
                continue

            point = (
                (box[0] + box[2]) // 2,
                (box[1] + box[3]) // 2,
            )
            if id in self.tracks:
                self.tracks[id].add_point(point)
                self.tracks[id].last_update = frame_i
            else:
                self.tracks[id] = Track([point], frame_i)

        remove = []
        for id, track in self.tracks.items():
            if frame_i - track.last_update >= 5:
                remove.append(id)
        for id in remove:
            self.tracks.pop(id)
