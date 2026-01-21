"""
PTZ control thread.
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO

from constants import *

yolo = YOLO("yolo26n.pt")


def detect_persons(frame):
    results = yolo(frame)
    results = results[0]

    # Filter by confidence and class.
    boxes = []
    for i in range(len(results.boxes)):
        box = results.boxes[i]
        cls_id = int(box.cls[0])
        conf = box.conf[0]
        if cls_id == 0 and conf >= CONF_THRES:
            boxes.append(box)

    return boxes


def annotate_frame(results, frame):
    for box in results:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def control_thread(state: ThreadState):
    # Current PTZ.
    curr_pos = np.array([0, 0, 0], dtype=int)

    while state.run:
        time.sleep(1 / CTRL_FPS)
        if len(state.frameq) == 0:
            continue
        frame = state.frameq[-1].copy()

        results = detect_persons(frame)

        annotate_frame(results, frame)
        cv2.imshow("a", frame)
        cv2.waitKey(1)

        state.camera.set(cv2.CAP_PROP_PAN, curr_pos[0])
        state.camera.set(cv2.CAP_PROP_TILT, curr_pos[1])
        state.camera.set(cv2.CAP_PROP_ZOOM, curr_pos[2])
