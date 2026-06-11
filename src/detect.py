"""
Person detecting using RT-DETR.
"""

import cv2
import numpy as np
import torch
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from field_mask import read_mask, create_mask, create_persp_scale
from utils import *

DETR_PROCESSOR = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
DETR_MODEL = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", device_map="auto")


class Detector:
    """
    Person detection with RT-DETR.
    Spectator filtering with manual field mask.

    The position of a player (wrt the field mask) is the midpoint of the bottom edge;
    i.e. where their feet are.
    This position is used to query the field mask.

    Players near the sideline are the most difficult to distinguish.
    First, we blur the field mask, to obtain a continuous value near the edge.
    """

    def __init__(self, field_mask_path):
        mask_points = read_mask(field_mask_path)
        self.field_mask = create_mask(mask_points).astype(np.float32)
        # Is a measure of closeness to border. -1 outside, 1 center, 0 on border.
        self.blurred_mask = cv2.blur(self.field_mask, (FIELD_MASK_BLUR, FIELD_MASK_BLUR))
        self.blurred_mask = 2 * self.blurred_mask - 1
        # Scale to account for far people being small. 1 near, 3 far.
        self.persp_scale = create_persp_scale(mask_points)

    def update(self, frame):
        """
        frame: cv2 format.
        motion_out: Output of Motion.update
        return: {
            boxes: ndarray float (N, 4) xyxy bounding boxes.
            filtered_boxes: Boxes of active players. Subset of bboxes.
        }
        """
        boxes = run_detr(frame).astype(int)
        filtered_boxes = self.filter_boxes(boxes)

        return {
            "boxes": boxes,
            "filtered_boxes": filtered_boxes,
            "blurred_mask": self.blurred_mask,
        }

    def filter_boxes(self, boxes):
        """
        Returns list of boxes that are active players.
        """
        ret = []
        for box in boxes:
            x1, y1, x2, y2 = box
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # TODO static thres for now
            if self.blurred_mask[y2, mid_x] * self.persp_scale[y2, mid_x] > 0.3:
                ret.append(box)

        ret = np.array(ret, dtype=np.float32)
        return ret


def run_detr(frame):
    """
    frame: cv2 format original frame.
    return: ndarray (N, 4) xyxy float bounding boxes.
    """
    # Run DETR.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = DETR_PROCESSOR(images=frame, return_tensors="pt").to(DEVICE)
    outputs = DETR_MODEL(**inputs)
    results = DETR_PROCESSOR.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([RES[::-1]]),
        threshold=0.2,
    )

    # Convert to bboxes.
    bboxes = []
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score, label = score.item(), label_id.item()
            if label == 0:
                bboxes.append(box.tolist())
    bboxes = np.array(bboxes, dtype=np.float32)

    return bboxes


def vis_detector(frame, detector_out):
    """
    frame: cv2 format original frame.
    detector_out: Dict output of Detector.update
    """
    frame = frame.copy()

    # Draw bboxes.
    for x1, y1, x2, y2 in detector_out["boxes"].astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x1, y1, x2, y2 in detector_out["filtered_boxes"].astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay field mask
    mask = detector_out["blurred_mask"] / 2 + 0.5
    mask = (mask * 255).astype(np.uint8)
    frame = cv2.addWeighted(frame, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow("Detector", frame)

    cv2.waitKey(1)


def vis_field_mask(mask):
    """
    mask: ndarray [H, W] bool
    """
    vis = (mask * 255).astype(np.uint8)
    cv2.imshow("Field Mask", vis)
    cv2.waitKey(0)
