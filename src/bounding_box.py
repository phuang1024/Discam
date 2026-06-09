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


def extract_boxes(detector_out):
    """
    Extract overall bounding box given person detections.
    detector_out: List of dicts.
        Each dict is output of Detector.update
    return: ndarray float (N, 4) xyxy
        Boxes corresponding to each element of detector_out.
    """
    boxes = []
    for data in detector_out:
        # Find min and max coords.
        xs = []
        ys = []
        for box in data["filtered_boxes"]:
            xs.append(int((box[0] + box[2]) / 2))
            ys.append(int((box[1] + box[3]) / 2))

        if len(xs) == 0 or len(ys) == 0:
            x1 = x2 = y1 = y2 = 0
        else:
            x1 = min(xs) - OUT_PADDING
            x2 = max(xs) + OUT_PADDING
            y1 = min(ys) - OUT_PADDING
            y2 = max(ys) + OUT_PADDING

        boxes.append((x1, y1, x2, y2))

    boxes = np.array(boxes, dtype=float)
    return boxes


def lerp_boxes(in_boxes, in_frames, frame_count):
    """
    Linear interpolation between boxes at frame intervals.
    in_boxes, in_frames: Output of extract_boxes
    frame_count: Total number of frames in output video.
    return: ndarray float (frame_count, 4) xyxy
    """
    # Current frame is between B[i] and B[i+1].
    in_index = 0

    ret = []
    for frame in range(frame_count):
        # Before first box.
        if frame <= in_frames[0]:
            ret.append(in_boxes[0])
            continue
        # After last box.
        if frame >= in_frames[-1] or in_index >= len(in_boxes) - 1:
            ret.append(in_boxes[-1])
            continue

        # Calculate lerp.
        fac = (frame - in_frames[in_index]) / (in_frames[in_index+1] - in_frames[in_index])
        box = (1 - fac) * in_boxes[in_index] + fac * in_boxes[in_index+1]
        ret.append(box)

        # Advance index.
        if frame > in_frames[in_index + 1]:
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


def ema_smooth_boxes(in_boxes):
    """
    Apply EMA variant.
    in_boxes: (N, 4)
    return: Same format.
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
    boxes: (N, 4)
    return: Same format.
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
    bbox: xyxy
    return: xyxy, ndarray float
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
    new_bbox = np.array([x1, y1, x2, y2], dtype=float)

    return new_bbox


def compute_final_boxes(detector_out, frame_count, out_fps):
    """
    Main function to call.

    in_boxes: Output of pipeline. List of dicts.
        Assumes sorted by frame number.
    frame_count: Total number of frames (i.e. ending frame number).
    return: List of bboxes for all frames.
        Each box is xyxy tuple of ints.
        First element is bbox for the first frame, etc..
    """
    boxes = extract_boxes(detector_out)
    # Frame numbers in output video coords.
    frames = np.arange(len(boxes)) * out_fps / FPS
    boxes = lerp_boxes(boxes, frames, frame_count)
    boxes = ema_smooth_boxes(boxes)
    for i in range(len(boxes)):
        boxes[i] = resize_bbox(boxes[i])
    boxes = moving_average(boxes)
    return boxes


def vis_static_bbox(frame, bbox_out):
    """
    frame: cv2 format
    bbox_out: Dict output of StaticBBox.update.
    """
    frame = frame.copy()

    # Draw box.
    box = bbox_out["bbox"]
    box = [int(x) for x in box]
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("StaticBBox", frame)
    cv2.waitKey(1)
