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
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=0)

        boxes = run_detr_tiled(frame).astype(int)
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
            if self.blurred_mask[y2, mid_x] * self.persp_scale[y2, mid_x] > 0.4:
                ret.append(box)

        ret = np.array(ret, dtype=np.float32)
        return ret


def run_detr_single(frame):
    """
    Run on single frame. Return person boxes.
    frame: cv2 format original frame.
    return: ndarray (N, 4) xyxy float bounding boxes.
    """
    # Run DETR.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = DETR_PROCESSOR(images=frame, return_tensors="pt").to(DEVICE)
    outputs = DETR_MODEL(**inputs)
    results = DETR_PROCESSOR.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[frame.shape[0], frame.shape[1]]]),
        threshold=0.3,
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


def run_detr_tiled(frame):
    """
    2x2 tiled inference.
    Expand each tile a bit. Remove detections near edge.
    frame: cv2 format.
    return: ndarray (N, 4) xyxy
        In coords of frame.
    """
    half_w = frame.shape[1] // 2
    half_h = frame.shape[0] // 2

    all_boxes = []
    for x in range(2):
        for y in range(2):
            # Find tile coords.
            x1 = x * half_w
            y1 = y * half_h
            x2 = (x + 1) * half_w
            y2 = (y + 1) * half_h
            # Expand.
            x1e = np.clip(x1 - 50, 0, frame.shape[1])
            y1e = np.clip(y1 - 50, 0, frame.shape[0])
            x2e = np.clip(x2 + 50, 0, frame.shape[1])
            y2e = np.clip(y2 + 50, 0, frame.shape[0])

            tile = frame[y1e:y2e, x1e:x2e]
            boxes = run_detr_single(tile)
            # Remove boxes near edge.
            for bx1, by1, bx2, by2 in boxes:
                if (bx1 < 10
                    or by1 < 10
                    or (tile.shape[1] - bx2) < 10
                    or (tile.shape[0] - by2) < 10):
                    continue
                all_boxes.append((bx1 + x1e, by1 + y1e, bx2 + x1e, by2 + y1e))

    boxes = np.array(all_boxes, dtype=np.float32)
    return boxes


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
