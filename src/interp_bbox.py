"""
Interpolate bbox.
Pipeline outputs bounding boxes every N frames.
This module:
1. Ensures correct aspect ratio and min size.
2. Interp between given bboxes.
"""

import numpy as np

from utils import *


def resize_bbox(bbox):
    """
    Resize to satisfy aspect and min size.
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Min size.
    width = max(width, BBOX_MIN_SIZE)
    height = max(height, BBOX_MIN_SIZE)

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
    return new_bbox


def lerp_bboxes(in_boxes, frame_count):
    """
    Linear interpolation between successive given bboxes.
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
        #ret.append(box)
        ret.append(box1["static_bbox"])

        # Advance index.
        if frame > in_boxes[in_index + 1]["frame_i"]:
            in_index += 1

    return ret


def interp_bboxes(in_boxes, frame_count):
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
    for box in in_boxes:
        box["static_bbox"] = np.array(resize_bbox(box["static_bbox"]))

    #boxes_lerp = lerp_bboxes(in_boxes, frame_count)

    ret = []
    #for box in boxes_lerp:
    #    ret.append(tuple(box.astype(int)))
    for box in in_boxes:
        ret.append(tuple(box["static_bbox"].astype(int)))
    return ret
