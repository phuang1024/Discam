"""
Interpolate bbox.
Pipeline outputs bounding boxes every N frames.
This module:
1. Ensures correct aspect ratio and min size.
2. Interp between given bboxes.

Filters:
1. Linear interp between boxes from pipeline.
2. EMA with different facs for expand vs shrink.
3. Aspect correction.
4. Moving average.
"""

import numpy as np

from utils import *


def lerp_bboxes(in_boxes, frame_count):
    """
    Linear interpolation between successive given bboxes.
    in_boxes: Same as for interp_bboxes
    return: List of ndarrays, one per frame.
    """
    # Current frame is between B[i] and B[i+1].
    in_index = 0

    ret = []
    for frame in range(frame_count):
        # Before first box.
        if frame <= in_boxes[0]["frame_i"]:
            ret.append(in_boxes[0]["static_bbox"])
            continue
        # After last box.
        if frame >= in_boxes[-1]["frame_i"] or in_index >= len(in_boxes) - 1:
            ret.append(in_boxes[-1]["static_bbox"])
            continue

        # Calculate lerp.
        box1 = in_boxes[in_index]
        box2 = in_boxes[in_index + 1]
        fac = (frame - box1["frame_i"]) / (box2["frame_i"] - box1["frame_i"])
        box = (1 - fac) * box1["static_bbox"] + fac * box2["static_bbox"]
        ret.append(box)

        # Advance index.
        if frame > in_boxes[in_index + 1]["frame_i"]:
            in_index += 1

    return ret


class SmoothEMA:
    """
    Smooth a scalar value over time.
    The value is one of the xyxy box coords.
    When the box is expanding, more responsive.
    When shrinking, less responsive and minimum margin.
    """

    def __init__(self):
        self.ema_value = None

    def update(self, value):
        if self.ema_value is None:
            self.ema_value = value
            return value

        if value > self.ema_value:
            # Expanding.
            self.ema_value = OUT_EXPAND_EMA * value + (1 - OUT_EXPAND_EMA) * self.ema_value
        else:
            value = min(value + OUT_SHRINK_MARGIN, self.ema_value)
            self.ema_value = OUT_SHRINK_EMA * value + (1 - OUT_SHRINK_EMA) * self.ema_value

        return self.ema_value


def smooth_bboxes(in_boxes):
    """
    Apply EMA variant.
    in_boxes: Output of lerp_bboxes
    return: Same format as in_boxes
    """
    x1_ema = SmoothEMA()
    y1_ema = SmoothEMA()
    x2_ema = SmoothEMA()
    y2_ema = SmoothEMA()

    ret = []
    for x1, y1, x2, y2 in in_boxes:
        x1 = -x1_ema.update(-x1)
        y1 = -y1_ema.update(-y1)
        x2 = x2_ema.update(x2)
        y2 = y2_ema.update(y2)
        ret.append((x1, y1, x2, y2))

    ret = np.array(ret, dtype=float)
    return ret


def moving_average(boxes, k=OUT_MOVING_AVG):
    """
    boxes: ndarray float (N, 4)
    return: Same as boxes.
    """
    ret = []
    moment = np.zeros(4, dtype=float)
    num_elements = 0
    for frame in range(len(boxes) + k):
        if frame < len(boxes):
            moment += boxes[frame]
            num_elements += 1
        if frame >= k:
            moment -= boxes[frame - k]
            num_elements -= 1
        if frame >= k - 1 and len(ret) < len(boxes):
            ret.append(moment / num_elements)

    return ret


def resize_bbox(bbox):
    """
    Resize to satisfy aspect, min size, and in bounds.
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Min size.
    width = max(width, OUT_MIN_SIZE)
    height = max(height, OUT_MIN_SIZE)

    # Aspect: Expand one of width or height.
    aspect = width / height
    if aspect > OUT_ASPECT:
        height = width / OUT_ASPECT
    else:
        width = height * OUT_ASPECT

    new_bbox = (
        int(cx - width / 2),
        int(cy - height / 2),
        int(cx + width / 2),
        int(cy + height / 2),
    )

    # In bounds.
    x1, y1, x2, y2 = new_bbox
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 >= RES[0]:
        x1 -= (x2 - RES[0] + 1)
        x2 = RES[0] - 1
    if y2 >= RES[1]:
        y1 -= (y2 - RES[1] + 1)
        y2 = RES[1] - 1
    new_bbox = (x1, y1, x2, y2)

    return new_bbox


def interp_bboxes(in_boxes, frame_count, out_fps):
    """
    Main function to call.
    Handles resizing and interpolation.

    in_boxes: Output of pipeline. List of dicts.
        Assumes sorted by frame number.
    frame_count: Total number of frames (i.e. ending frame number).
    return: List of bboxes for all frames.
        Each box is xyxy tuple of ints.
        First element is bbox for the first frame, etc..
    """
    # Change frame number to be in output video coords.
    for box in in_boxes:
        box["static_bbox"] = np.array(box["static_bbox"], dtype=float)
        box["frame_i"] = box["frame_i"] * out_fps / FPS

    boxes = lerp_bboxes(in_boxes, frame_count)
    boxes = smooth_bboxes(boxes)
    for i in range(len(boxes)):
        boxes[i] = resize_bbox(boxes[i])
    boxes = moving_average(boxes)
    return boxes
