"""
Utils for using YOLO to detect people.
"""

import cv2
from ultralytics import YOLO

yolo = YOLO("yolo12m.pt")


def vis_yolo(frame, mask, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        in_roi = mask[y1:y2, x1:x2].any()
        color = (0, 255, 0) if in_roi else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


def detect_people(frame, mask):
    """
    Compute bounding box of a single frame.
    """
    # Class 0 is person
    detections = yolo(frame, conf=0.05, classes=[0])
    boxes = detections[0].boxes

    vis_yolo(frame, mask, boxes)
