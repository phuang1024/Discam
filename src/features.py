"""
ORB feature extraction and matching.

YOLO object detection.
"""

import cv2
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
