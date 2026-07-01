"""Compute and filter crop boxes from CV outputs.
"""

import cv2
import numpy as np

from ..utils.constants import *


def extract_box(detections, padding=BOX_PADDING):
    """Extract overall bounding box given person detections.

    Args:
        detections: ``boxes format``.
        padding: Min space between frame and outermost person.
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


def lerp_boxes(in_boxes, frame_is, frame_count):
    """Linear interpolation between boxes.
    CV pipeline is run at a lower FPS than input video.

    Args:
        in_boxes: ``boxes format``, crop boxes from ``extract_box``.
        frame_is: ``list int (N,)``, frame numbers of each box.
        frame_count: Total number of frames in input video.

    Returns:
        ``boxes format``. Length ``frame_count`` (longer than input).
    """
    # Means current frame is between B[i] and B[i+1].
    in_index = 0

    ret = []
    for frame in range(frame_count):
        # Before first box.
        if frame <= frame_is[0]:
            ret.append(in_boxes[0])
            continue
        # After last box.
        if frame >= frame_is[-1] or in_index >= len(in_boxes) - 1:
            ret.append(in_boxes[-1])
            continue

        # Calculate lerp.
        fac = (frame - frame_is[in_index]) / (frame_is[in_index+1] - frame_is[in_index])
        box = (1 - fac) * in_boxes[in_index] + fac * in_boxes[in_index+1]
        ret.append(box)

        # Advance in_boxes pointer.
        if frame > frame_is[in_index + 1]:
            in_index += 1

    return ret


class SmoothEMA:
    """EMA variant. See ``ema_smooth_boxes``.
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
    """Apply the EMA variant filter.
    More responsive when increasing, to not lose footage.
    Less responsive and minimum margin when decreasing, to be less jittery.

    Args:
        in_boxes: ``boxes format``.

    Returns:
        Same shape.
    """
    x1_ema = SmoothEMA()
    y1_ema = SmoothEMA()
    x2_ema = SmoothEMA()
    y2_ema = SmoothEMA()

    ret = np.zeros_like(in_boxes)
    for i, (x1, y1, x2, y2) in enumerate(in_boxes):
        # Reverse x1 and y1 "increase" and "decrease".
        x1 = -x1_ema.update(-x1)
        y1 = -y1_ema.update(-y1)
        x2 = x2_ema.update(x2)
        y2 = y2_ema.update(y2)
        ret[i] = (x1, y1, x2, y2)
    return ret


def resize_box(box, target_aspect):
    """Resize to satisfy:

    - Target aspect ratio.
    - Minimum dimensions.
    - In bounds of original image.

    Args:
        box: A single ``xyxy`` box.
        target_aspect: Target ``W/H`` aspect ratio.

    Returns:
        Same format as ``box``.
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
    if aspect > target_aspect:
        height = width / target_aspect
    else:
        width = height * target_aspect
    x1 = int(cx - width / 2)
    y1 = int(cy - height / 2)
    x2 = int(cx + width / 2)
    y2 = int(cy + height / 2)

    # Check in bounds.
    if x2 - x1 > CV_RES[0] or y2 - y1 > CV_RES[1]:
        return [0, 0, CV_RES[0], CV_RES[1]]
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 >= CV_RES[0]:
        x1 -= (x2 - CV_RES[0] + 1)
        x2 = CV_RES[0] - 1
    if y2 >= CV_RES[1]:
        y1 -= (y2 - CV_RES[1] + 1)
        y2 = CV_RES[1] - 1

    return (x1, y1, x2, y2)


def moving_average(boxes, k=BOX_MOVING_AVG):
    """Apply moving average filter on each coord independently.

    Args:
        boxes: ``boxes format``.
        k: Window size.

    Returns:
        Same format.
    """
    ret = []
    # Window sum.
    moment = np.zeros(4, dtype=float)
    num_elements = 0
    for i in range(len(boxes) + k):
        if i < len(boxes):
            moment += boxes[i]
            num_elements += 1
        if i >= k:
            moment -= boxes[i - k]
            num_elements -= 1
        if i >= k - 1 and len(ret) < len(boxes):
            ret.append(moment / num_elements)

    ret = np.array(ret, dtype=float)
    return ret


def compute_final_boxes(pipe_outs, frame_is, frame_count):
    """Main function to convert CV outputs into crop boxes for Post Processing.

    - Extract box per frame with active person detections.
    - Linear interp between boxes for intermediate frames.
    - EMA with different facs for expand vs shrink.
    - Aspect, size, and bounds correction.
    - Large window moving average.

    Args:
        pipe_outs: List of CV pipeline outputs.
        frame_is: List of frame indices the pipe outputs correspond to, in input video coords.
        frame_count: Total number of frames in input video.

    Returns:
        ``boxes format``, length `frame_count`.
        Box for each frame in input video.
    """
    # Extract boxes.
    boxes = []
    for data in pipe_outs:
        box = extract_box(data["active_boxes"])
        if box is None:
            box = boxes[-1] if boxes else (0, 0, OUT_RES[0], OUT_RES[1])
        boxes.append(box)
    boxes = np.array(boxes, dtype=int)

    boxes = lerp_boxes(boxes, frame_is, frame_count)

    boxes = ema_smooth_boxes(boxes)

    aspect = OUT_RES[0] / OUT_RES[1]
    for i in range(len(boxes)):
        boxes[i] = resize_box(boxes[i], aspect)

    boxes = moving_average(boxes)
    return boxes


def vis_static_box(frame, box):
    """Visualize crop box on frame.

    Args:
        frame: ``cv2 format``.
        box: xyxy.
    """
    frame = frame.copy()
    # Draw box.
    box = [int(x) for x in box]
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Bounding box", frame)
    cv2.waitKey(1)
