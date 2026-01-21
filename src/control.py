"""
PTZ control thread.
"""

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from constants import *

yolo = YOLO("yolo26n.pt")


@dataclass
class ResultsAverage:
    """
    Queues to handle moving window average of bbox results.
    """
    # Number of bboxes.
    n: deque
    x_moment: deque
    y_moment: deque
    size_moment: deque

    def __init__(self, max_len):
        self.n = deque(maxlen=max_len)
        self.x_moment = deque(maxlen=max_len)
        self.y_moment = deque(maxlen=max_len)
        self.size_moment = deque(maxlen=max_len)

    def get_avg(self):
        total_n = sum(self.n)
        return (
            sum(self.x_moment) / total_n,
            sum(self.y_moment) / total_n,
            sum(self.size_moment) / total_n,
        )

    def update_new_results(self, results):
        """
        results: Return from detect_persons().
        """
        x_moment = 0
        y_moment = 0
        size_moment = 0
        for box in results:
            x1, y1, x2, y2 = box.xyxy[0]
            x_moment += (x1 + x2) / 2
            y_moment += (y1 + y2) / 2
            size_moment += y2 - y1

        self.n.append(len(results))
        self.x_moment.append(x_moment)
        self.y_moment.append(y_moment)
        self.size_moment.append(size_moment)


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


def annotate_frame(results, avg, frame):
    for box in results:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.circle(frame, (int(avg[0]), int(avg[1])), 5, (255, 0, 0), -1)


def control_thread(state: ThreadState):
    # Current PTZ.
    curr_pos = np.array([0, 0, 0], dtype=int)

    avg_results = ResultsAverage(CTRL_AVG_WINDOW)

    while state.run:
        time.sleep(1 / CTRL_FPS)
        if len(state.frameq) == 0:
            continue
        frame = state.frameq[-1].copy()

        results = detect_persons(frame)
        avg_results.update_new_results(results)
        avg = avg_results.get_avg()

        annotate_frame(results, avg, frame)
        cv2.imshow("a", frame)
        cv2.waitKey(1)

        state.camera.set(cv2.CAP_PROP_PAN, curr_pos[0])
        state.camera.set(cv2.CAP_PROP_TILT, curr_pos[1])
        state.camera.set(cv2.CAP_PROP_ZOOM, curr_pos[2])
