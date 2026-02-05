"""
PTZ control thread.
"""

import json
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
    # Center X coords of each bbox.
    xs: deque
    ys: deque
    sizes: deque

    def __init__(self, max_len):
        self.xs = deque(maxlen=max_len)
        self.ys = deque(maxlen=max_len)
        self.sizes = deque(maxlen=max_len)

    def update(self, detects):
        """
        detects: Return from detect_persons().
        """
        xs = []
        ys = []
        sizes = []
        for box in detects:
            x1, y1, x2, y2 = box.xyxy[0]
            xs.append((x1 + x2) / 2)
            ys.append((y1 + y2) / 2)
            sizes.append(y2 - y1)

        self.xs.append(xs)
        self.ys.append(ys)
        self.sizes.append(sizes)

    def get_avg(self):
        """
        If no bboxes in the entire queue, returns None.
        return {
            n: Average bboxes per frame.
            median_box: A new xyxy bounding box, using the 25 and 75 percentiles of
                the centers of all detection bboxes.
        }
        """
        total_n = sum(len(x) for x in self.xs)
        if total_n == 0:
            return None

        xs = np.concatenate(self.xs)
        ys = np.concatenate(self.ys)
        x_min = np.quantile(xs, BOX_QUANTILE).item()
        x_max = np.quantile(xs, 1 - BOX_QUANTILE).item()
        y_min = np.quantile(ys, BOX_QUANTILE).item()
        y_max = np.quantile(ys, 1 - BOX_QUANTILE).item()
        median_box = [x_min, y_min, x_max, y_max]

        return {
            "n": total_n / len(self.xs),
            "median_box": median_box,
        }


def detect_persons(frame):
    detects = yolo(frame)
    detects = detects[0]

    # Filter by confidence and class.
    boxes = []
    for i in range(len(detects.boxes)):
        box = detects.boxes[i]
        cls_id = int(box.cls[0])
        conf = box.conf[0]
        if cls_id == 0 and conf >= CONF_THRES:
            boxes.append(box)

    return boxes


def annotate_frame(detects, avg, frame):
    for box in detects:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if avg is not None:
        x1, y1, x2, y2 = map(int, avg["median_box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


def control_thread(state: ThreadState, log_dir):
    # Current PTZ.
    curr_pos = np.array([0, 0, 0], dtype=int)

    averager = ResultsAverage(CTRL_AVG_WINDOW)

    index = 0

    while state.run:
        time.sleep(1 / CTRL_FPS)
        if len(state.frameq) == 0:
            continue
        frame = state.frameq[-1].copy()

        detects = detect_persons(frame)
        averager.update(detects)
        avg = averager.get_avg()

        annotate_frame(detects, avg, frame)
        cv2.imshow("a", frame)
        cv2.waitKey(1)

        state.camera.set(cv2.CAP_PROP_PAN, curr_pos[0])
        state.camera.set(cv2.CAP_PROP_TILT, curr_pos[1])
        state.camera.set(cv2.CAP_PROP_ZOOM, curr_pos[2])

        log_step(log_dir, index, frame, detects, avg)
        index += 1


def log_step(dir, index, frame, detects, avg):
    """
    Log control results to disk.
    """
    cv2.imwrite(str(dir / f"{index}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

    bboxes = [box.xyxy.tolist() for box in detects]
    with open(dir / f"{index}.detects.json", "w") as fp:
        json.dump(bboxes, fp, indent=4)

    with open(dir / "{index}.avg.json", "w") as fp:
        json.dump(avg, fp, indent=4)
