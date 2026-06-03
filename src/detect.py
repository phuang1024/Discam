"""
Person detecting using RT-DETR
"""

import cv2
import numpy as np
import torch
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from utils import *

DETR_PROCESSOR = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
DETR_MODEL = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", device_map="auto")


class Detector:
    def __init__(self):
        pass

    def update(self, frame):
        """
        frame: cv2 format.
        return: {
            bboxes: (N, 4) xyxy float bounding boxes.
        }
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

        return {
            "bboxes": bboxes,
        }


def vis_detector(frame, detector_out):
    """
    frame: cv2 format original frame.
    detector_out: Dict output of Detector.update
    """
    frame = frame.copy()

    # Draw bboxes.
    for box in detector_out["bboxes"]:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)
