"""
Person detecting using RT-DETR
"""

import cv2
import numpy as np
import torch
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from field_mask import read_mask, create_mask
from utils import *

DETR_PROCESSOR = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
DETR_MODEL = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", device_map="auto")


class Detector:
    """
    Person detection with RT-DETR.
    Spectator filtering with manual field mask
    and other techniques.

    The position of a player (wrt the field mask) is the midpoint of the bottom edge;
    i.e. where their feet are.
    Note that however, the entire box is used to query OF.

    Players near the sideline are the most difficult to distinguish.
    First, we blur the field mask, to obtain a continuous value near the edge.

    Using OF, predict the near future movement for each player; i.e. r + v*dt
    If this lies inside the field, they are probably active.
    """

    def __init__(self, field_mask_path):
        self.field_mask = None
        if field_mask_path is not None:
            self.field_mask = create_mask(read_mask(field_mask_path)).astype(np.float32)
            self.blurred_mask = cv2.blur(self.field_mask, (FIELD_MASK_BLUR, FIELD_MASK_BLUR))

    def update(self, frame, motion_out):
        """
        frame: cv2 format.
        motion_out: Output of Motion.update
        return: {
            boxes: ndarray float (N, 4) xyxy bounding boxes.
            filtered_boxes: Boxes of active players. Subset of bboxes.
        }
        """
        boxes = run_detr(frame).astype(int)
        filtered_boxes = self.filter_boxes(boxes, motion_out["of"])

        return {
            "boxes": boxes,
            "filtered_boxes": filtered_boxes,
            "blurred_mask": self.blurred_mask,
        }

    def filter_boxes(self, boxes, of):
        """
        Returns list of boxes that are active players.
        of: ndarray float (H, W, 2) optical flow.
        """
        of_mag = np.linalg.norm(of, axis=-1)

        ret = []
        for box in boxes:
            x1, y1, x2, y2 = box
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # TODO static thres for now
            if self.blurred_mask[y2, mid_x] > 0.6:
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


def vis_detector(frame, detector_out, motion_out):
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

    # Draw motion line on each person.
    of = motion_out["of"]
    for x1, y1, x2, y2 in detector_out["boxes"].astype(int):
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        velocity = of[mid_y, mid_x]
        p2 = (int(mid_x + velocity[0] * 3), int(mid_y + velocity[1] * 3))
        cv2.line(frame, (mid_x, mid_y), p2, (255, 0, 0), 2)

    mask_overlay = (detector_out["blurred_mask"] * 255).astype(np.uint8)
    frame = cv2.addWeighted(frame, 1.0, cv2.cvtColor(mask_overlay, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow("Detector", frame)

    cv2.waitKey(1)


def vis_field_mask(mask):
    """
    mask: ndarray [H, W] bool
    """
    vis = (mask * 255).astype(np.uint8)
    cv2.imshow("Field Mask", vis)
    cv2.waitKey(0)
