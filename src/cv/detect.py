"""
Person detection and tracking module.
"""

import cv2
import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm

from utils.constants import *

YOLO = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.2,
)


class Detector:
    """
    Detection with YOLO and SAHI. Tracking with TODO.
    """

    def __init__(self):
        pass

    def update(self, frame):
        """
        frame: cv2 format.
        return: boxes format.
            Boxes of all detected people.
        """
        results = get_sliced_prediction(
            frame,
            YOLO,
            slice_height=500,
            slice_width=500,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )
        # Convert to boxes format.
        boxes = []
        for r in results.object_prediction_list:
            if r.category.id == 0 and r.score > 0.2:
                boxes.append(r.bbox.to_xyxy())
        boxes = np.array(boxes, dtype=int)
        vis_detector(frame, boxes)
        return boxes


def post_run_detector(video_path):
    """
    Run detector on video file.
    Respects FPS and RES setting.
    return: List of {
        "frame_i": Frame index in original video coord.
        "boxes": Detector.update()
    }
    """
    video = cv2.VideoCapture(video_path)
    orig_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = int(video.get(cv2.CAP_PROP_FPS))
    orig_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_scale = int(orig_fps / DET_FPS)

    detector = Detector()

    outputs = []
    frame_i = 0
    pbar = tqdm(total=orig_len // fps_scale, desc="Detector")
    while True:
        for _ in range(fps_scale):
            ret, frame = video.read()
        frame_i += fps_scale
        if not ret:
            break

        if orig_w != DET_RES[0] or orig_h != DET_RES[1]:
            frame = cv2.resize(frame, DET_RES)

        outputs.append({
            "frame_i": frame_i,
            "boxes": detector.update(frame),
        })
        pbar.update(1)

    pbar.close()
    video.release()
    return outputs


def vis_detector(frame, player_boxes):
    frame = frame.copy()

    # Draw bboxes.
    """
    for x1, y1, x2, y2 in detector_out["boxes"].astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    """
    for x1, y1, x2, y2 in player_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay field mask
    """
    mask = detector_out["blurred_mask"] / 2 + 0.5
    mask = (mask * 255).astype(np.uint8)
    frame = cv2.addWeighted(frame, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    """

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)
