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
    Person detection with RT-DETR, manual field mask,
    and various post-processing techniques.

    RT-DETR: Returns bboxes of people.

    Field mask: Manual polygonal mask of where the field is.
        Check if the lower edge of bbox is in field.

    Inactive filter: Try to filter out spectators close to the field.
        We keep a "occupancy" map:
            Players' bboxes gradually increase magnitude in that location.
            Entire map slowly decreases.
        We consider occupancy map most strongly near the edges of the field mask
            (where spectators probably are).
        If the occupancy gets too high, we don't consider bboxes in that location.
    """

    def __init__(self, field_mask_path):
        self.occupancy_map = np.zeros(RES[::-1], dtype=np.float32)

        self.field_mask = None
        if field_mask_path is not None:
            self.field_mask = create_mask(read_mask(field_mask_path))
            self.blurred_mask = self.blur_field_mask(self.field_mask)

    def update(self, frame):
        """
        frame: cv2 format.
        return: {
            occupancy: (H, W) float occupancy map.
            bboxes: (N, 4) xyxy float bounding boxes.
            filtered_bboxes: Remaining valid bboxes after field mask and occupancy filter.
        }
        """
        bboxes = run_detr(frame).astype(int)

        # Update occupancy map.
        bboxes_mask = np.zeros(RES[::-1], dtype=np.float32)
        for box in bboxes:
            bboxes_mask[box[1] : box[3], box[0] : box[2]] = 1

        # EMA increase with bboxes.
        self.occupancy_map = self.occupancy_map * (1 - OCCU_INC_FAC) + bboxes_mask * OCCU_INC_FAC
        # Exponential decrease.
        self.occupancy_map = self.occupancy_map * (1 - OCCU_DEC_FAC)

        # Check: If box bottom edge is in mask,
        # and if occupancy * (1 - blurred_mask) is less than thres.
        filtered_bboxes = bboxes
        spectator_map = None
        if self.field_mask is not None:
            spectator_map = self.occupancy_map * (1 - self.blurred_mask)

            filtered_bboxes = []
            for box in bboxes:
                bottom_x = (box[0] + box[2]) // 2
                bottom_y = box[3]
                bottom_x = np.clip(bottom_x, 0, RES[0] - 1)
                bottom_y = np.clip(bottom_y, 0, RES[1] - 1)

                in_field = self.field_mask[bottom_y, bottom_x] > 0.5
                is_spectator = spectator_map[bottom_y - 2, bottom_x] > SPECTATOR_THRES
                if in_field and not is_spectator:
                    filtered_bboxes.append(box)

            filtered_bboxes = np.array(filtered_bboxes, dtype=float)

        return {
            "occupancy": self.occupancy_map,
            "spectator": spectator_map,
            "bboxes": bboxes,
            "filtered_bboxes": filtered_bboxes,
        }

    def blur_field_mask(self, mask):
        """
        mask: ndarray [H, W] float in [0, 1]
        return: blurred mask.
        """
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, None, fx=1/FIELD_MASK_BLUR, fy=1/FIELD_MASK_BLUR)
        mask = cv2.blur(mask, (7, 7))
        mask = cv2.resize(mask, None, fx=FIELD_MASK_BLUR, fy=FIELD_MASK_BLUR, interpolation=cv2.INTER_LINEAR)
        mask = mask.astype(float) / 255
        return mask


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

    # Show occupancy map.
    occupancy = (detector_out["occupancy"] * 255).astype(np.uint8)
    cv2.imshow("Occupancy", occupancy)

    # Show spectator map.
    spectator = (detector_out["spectator"] * 255).astype(np.uint8)
    cv2.imshow("spectator", spectator)

    # Draw bboxes.
    for box in detector_out["bboxes"]:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for box in detector_out["filtered_bboxes"]:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)


def vis_field_mask(mask):
    """
    mask: ndarray [H, W] bool
    """
    vis = (mask * 255).astype(np.uint8)
    cv2.imshow("Field Mask", vis)
    cv2.waitKey(0)
