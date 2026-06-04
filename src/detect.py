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

    def update(self, frame):
        """
        frame: cv2 format.
        return: {
            occupancy: (H, W) float occupancy map.
            bboxes: (N, 4) xyxy float bounding boxes.
            filtered_bboxes: Remaining valid bboxes after field mask and occupancy filter.
        }
        """
        bboxes = run_detr(frame)

        # Check if their feet is in field mask.
        if self.field_mask is not None:
            filtered_bboxes = []
            for box in bboxes:
                bottom_x = int((box[0] + box[2]) / 2)
                bottom_y = int(box[3])
                bottom_x = np.clip(bottom_x, 0, self.field_mask.shape[1] - 1)
                bottom_y = np.clip(bottom_y, 0, self.field_mask.shape[0] - 1)
                if self.field_mask[bottom_y, bottom_x] > 0:
                    filtered_bboxes.append(box)
            filtered_bboxes = np.array(filtered_bboxes, dtype=float)

        else:
            filtered_bboxes = bboxes

        return {
            "occupancy": self.occupancy_map,
            "bboxes": bboxes,
            "filtered_bboxes": filtered_bboxes,
        }


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
    for box in detector_out["bboxes"]:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for box in detector_out["filtered_bboxes"]:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)
