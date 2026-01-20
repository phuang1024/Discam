"""
Person tracking via YOLO.
"""

from ultralytics import YOLO

yolo = YOLO("yolo26n.pt")


def detect_people(frame):
    """
    frame: RGB frame.
    """
    results = yolo(frame)
