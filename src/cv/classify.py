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
    """
    points: list
    """List of (x, y) box lower edge points."""
    last_update: int
    """Frame number last updated."""


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

        self.tracks = {}

    def update(self, boxes, frame_i):
        """
        boxes: tracked boxes format. From detector.
        return: List of indices of `boxes` that are active players.
        """
        self.update_tracks(boxes, frame_i)
        return self.filter_mask_thres(boxes, 0.8)

    def filter_mask_thres(self, boxes, thres):
        """
        Query field mask at bottom edge of each box.
        Return list of indices of boxes with mask value greater than thres.
        """
        indices = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            mid_x = (x1 + x2) // 2
            if self.field_mask[y2, mid_x] > thres:
                indices.append(i)
        return indices

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
                self.tracks[id].points.append(point)
                self.tracks[id].last_update = frame_i
            else:
                self.tracks[id] = Track([point], frame_i)

        remove = []
        for id, track in self.tracks.items():
            if frame_i - track.last_update >= 5:
                remove.append(id)
        for id in remove:
            self.tracks.pop(id)
