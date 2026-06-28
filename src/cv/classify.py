"""
Active player classification module.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from utils.constants import *
from utils.field_mask import create_mask


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

        #self.tracks = {}

    def update(self, boxes, frame_i):
        """
        boxes: tracked boxes format. From detector.
        return: ndarray bool, same length as `boxes`.
            Whether each box is active.
        """
        #self.update_tracks(boxes, frame_i)

        active_inds = np.zeros([len(boxes)], dtype=bool)
        for i, box in enumerate(boxes):
            # Query mask at lower edge of box.
            mask_value = self.field_mask[box[3], (box[0] + box[2]) // 2]
            active_inds[i] = mask_value > 0

            # Check if velocity is too low.
            """
            if box[4] in self.tracks:
                vel = self.tracks[box[4]].get_avg_vel()
                #vel = (vel + 1) / len(self.tracks[box[4]].points)
                if len(self.tracks[box[4]].points) >= 4 and vel < 5:
                    active = False
            """

        return active_inds

    def knn_median_filter(self, boxes, active_inds, k=5):
        """
        Apply KNN median filter to whether each box is active.
        For each box, find k nearest boxes (including itself).
            Then use majority decision.
        X and Y distance are weighted differently, to account for perspective.
            Additionally, apply perspective scaling.
        """

    def compute_distance(self, box1, box2):
        """
        Compute distance between lower edge centerpoints.
        Applies both XY and Persp scaling.
        """

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
