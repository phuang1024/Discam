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
            median_box: A new xyxy bounding box, using the X and 1-X percentiles of
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


def compute_control(avg):
    """
    Returns delta PTZ based on averaged detection results.
    """
    bbox = avg["median_box"]

    # Measurements as a factor.
    width = (bbox[2] - bbox[0]) / WIDTH
    x = (bbox[2] + bbox[0]) / 2 / WIDTH - 0.5
    y = (bbox[3] + bbox[1]) / 2 / HEIGHT - 0.5

    # Compute zoom
    zoom = 0
    if width < ZOOM_CENTER - ZOOM_THRES:
        zoom = int((ZOOM_CENTER - width - ZOOM_THRES) * ZOOM_SPEED)
    elif width > ZOOM_CENTER + ZOOM_THRES:
        zoom = int((ZOOM_CENTER - width + ZOOM_THRES) * ZOOM_SPEED)

    # Compute PT
    pan = 0
    if x < -PT_THRES:
        pan = int((x + PT_THRES) * PT_SPEED)
    elif x > PT_THRES:
        pan = int((x - PT_THRES) * PT_SPEED)

    tilt = 0
    if y < -PT_THRES:
        tilt = int((y + PT_THRES) * PT_SPEED)
    elif y > PT_THRES:
        tilt = int((y - PT_THRES) * PT_SPEED)
    tilt *= -1

    return (pan, tilt, zoom)


def control_thread(state: ThreadState, log_dir):
    # Current PTZ.
    curr_pos = np.array([0, 0, 0], dtype=int)
    averager = ResultsAverage(CTRL_AVG_WINDOW)
    index = 0
    last_ctrl = 0

    while state.run:
        time.sleep(1 / CTRL_FPS)
        if len(state.frameq) == 0:
            continue
        frame = state.frameq[-1].copy()

        # Run detection.
        detects = detect_persons(frame)
        averager.update(detects)
        avg = averager.get_avg()

        # Draw annotations.
        annotate_frame(detects, avg, frame)
        cv2.imshow("a", frame)
        cv2.waitKey(1)

        if avg is not None and time.time() - last_ctrl > CTRL_DELAY:
            ctrl = compute_control(avg)

            if ctrl[0] != 0 or ctrl[1] != 0 or ctrl[2] != 0:
                print("Applying control:", ctrl)
                curr_pos += ctrl
                # Constrain zoom.
                curr_pos[2] = max(min(curr_pos[2], ZOOM_MAX), 1)

                state.camera.set_ptz(*curr_pos)
                last_ctrl = time.time()

        # Logging.
        log_step(log_dir, index, frame, detects, avg, curr_pos)
        index += 1


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


def log_step(dir, index, frame, detects, avg, curr_pos):
    """
    Log control results to disk.
    """
    cv2.imwrite(str(dir / f"{index}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

    bboxes = [box.xyxy.tolist() for box in detects]
    with open(dir / f"{index}.detects.json", "w") as fp:
        json.dump(bboxes, fp, indent=4)

    with open(dir / "{index}.avg.json", "w") as fp:
        json.dump(avg, fp, indent=4)

    with open(dir / f"{index}.curr_pos.json", "w") as fp:
        json.dump(curr_pos.tolist(), fp, indent=4)
