"""
Compute overall bounding box given detections.
Also post processing smoothing.

Steps:
- Extract box per frame based on person detections.
- Temporal median filter.
- Linear interp between boxes for remaining frames.
- EMA with different facs for expand vs shrink.
- Aspect correction.
- Large window moving average.
"""

import cv2
import numpy as np

from utils.constants import *


def extract_box(detections, padding=BOX_PADDING):
    """
    Extract overall bounding box given person detections.
    detections: (N, 4) xyxy
    """
    # Find min and max coords.
    xs = []
    ys = []
    for box in detections:
        xs.append(int((box[0] + box[2]) / 2))
        ys.append(int((box[1] + box[3]) / 2))

    if len(xs) == 0 or len(ys) == 0:
        return None
    else:
        x1 = min(xs) - padding
        x2 = max(xs) + padding
        y1 = min(ys) - padding
        y2 = max(ys) + padding
        return x1, y1, x2, y2


def median_filter(boxes, k):
    """
    boxes: (N, 4)
    return: Same format.
    """
    ret = []
    for i in range(len(boxes)):
        start = max(0, i - k // 2)
        end = min(len(boxes), i + k // 2 + 1)
        median_box = np.median(boxes[start:end], axis=0)
        ret.append(median_box)

    return ret


def lerp_boxes(in_boxes, in_frames, frame_count):
    """
    Linear interpolation between boxes at frame intervals.
    in_boxes: boxes format.
    in_frames: List of frame numbers in output coordinates corresponding to each box.
    frame_count: Total number of frames in output video.
    return: boxes format. Length `frame_count`.
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
            self.ema_value = BOX_EXPAND_EMA * value + (1 - BOX_EXPAND_EMA) * self.ema_value
        else:
            value = min(value + BOX_SHRINK_MARGIN, self.ema_value)
            self.ema_value = BOX_SHRINK_EMA * value + (1 - BOX_SHRINK_EMA) * self.ema_value

        return self.ema_value


def ema_smooth_boxes(in_boxes):
    """
    Apply EMA variant filter.
    in_boxes: boxes format.
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


def moving_average(boxes, k=BOX_MOVING_AVG):
    """
    Apply moving average filter.
    boxes: boxes format.
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


def resize_box(box, out_aspect):
    """
    Resize to satisfy aspect, min size, and in bounds.
    box: xyxy
    out_aspect: Target W/H aspect ratio.
    return: xyxy, ndarray float
    """
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Min size.
    width = max(width, BOX_MIN_SIZE)
    height = max(height, BOX_MIN_SIZE)

    # Aspect: Expand one of width or height.
    aspect = width / height
    if aspect > out_aspect:
        height = width / out_aspect
    else:
        width = height * out_aspect
    x1 = int(cx - width / 2)
    y1 = int(cy - height / 2)
    x2 = int(cx + width / 2)
    y2 = int(cy + height / 2)

    # Check in bounds.
    if x2 - x1 > DET_RES[0] or y2 - y1 > DET_RES[1]:
        return [0, 0, DET_RES[0], DET_RES[1]]
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 >= DET_RES[0]:
        x1 -= (x2 - DET_RES[0] + 1)
        x2 = DET_RES[0] - 1
    if y2 >= DET_RES[1]:
        y1 -= (y2 - DET_RES[1] + 1)
        y2 = DET_RES[1] - 1

    return [x1, y1, x2, y2]


def compute_final_boxes(pipe_outs, frame_is, frame_count):
    """
    Main function to call.
    Converts pipeline outputs (list of active player detections)
    to a sequence of crop boxes for each frame, with filtering.

    pipe_outs: List of pipeline outputs.
    frame_is: List of frame indices the pipe outputs correspond to,
        in input video coords.
    frame_count: Total number of frames in output video.
    return: boxes format. Length `frame_count`.
        Corresponds to each frame in out video.
    """
    boxes = []
    for player_boxes in pipe_outs:
        box = extract_box(player_boxes)
        if box is None:
            box = boxes[-1] if boxes else (0, 0, OUT_RES[0], OUT_RES[1])
        boxes.append(box)
    boxes = np.array(boxes, dtype=int)

    boxes = median_filter(boxes, BOX_MEDIAN_FILTER)

    # In and out video FPS is same, so we can use frame_is as is.
    boxes = lerp_boxes(boxes, frame_is, frame_count)

    boxes = ema_smooth_boxes(boxes)

    out_aspect = OUT_RES[0] / OUT_RES[1]
    for i in range(len(boxes)):
        boxes[i] = resize_box(boxes[i], out_aspect)

    boxes = moving_average(boxes)
    return boxes


def vis_static_bbox(frame, box):
    """
    frame: cv2 format.
    box: xyxy.
    """
    frame = frame.copy()

    # Draw box.
    box = [int(x) for x in box]
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("StaticBBox", frame)
    cv2.waitKey(1)
