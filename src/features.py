"""
ORB feature extraction and matching.

YOLO object detection.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

yolo = YOLO("yolo11n.pt")

orb_detector = cv2.ORB_create()
orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def run_yolo(frame):
    preds = yolo.predict(source=frame)
    return preds


def run_orb(img1, img2):
    key1, des1 = orb_detector.detectAndCompute(img1, None)
    key2, des2 = orb_detector.detectAndCompute(img2, None)
    matches = orb_matcher.match(des1, des2)

    return (key1, des1), (key2, des2), matches


def get_keypoints(img1, img2, num_keypoints=30):
    """
    Returns two arrays, each shape (N, 2), of keypoint locations.
    Normalized to [0, 1].

    Takes first N keypoints, by min distance.
    """
    (key1, des1), (key2, des2), matches = run_orb(img1, img2)
    print(len(matches))

    # Take first N keypoints.
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:num_keypoints]

    from_pts = torch.empty((len(matches), 2))
    to_pts = torch.empty((len(matches), 2))
    for i, match in enumerate(matches):
        p1 = key1[match.queryIdx].pt
        p2 = key2[match.trainIdx].pt
        from_pts[i] = torch.tensor(p1)
        to_pts[i] = torch.tensor(p2)

    # Normalize to [0, 1].
    shape = np.array([img1.shape[1], img1.shape[0]])
    from_pts /= shape
    to_pts /= shape

    return from_pts, to_pts
