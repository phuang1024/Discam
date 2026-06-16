"""
Person detecting using RT-DETR.
Motion analysis using Farneback optical flow.
"""

import cv2
import numpy as np
import torch
from torchvision.utils import flow_to_image

from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from bounding_box import extract_box, resize_bbox
from field_mask import read_mask, create_mask, create_persp_scale
from utils import *

DETR_PROCESSOR = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
DETR_MODEL = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", device_map="auto")


class Detector:
    """
    Person detection with RT-DETR, and spectator classification.
    Motion analysis using optical flow.

    2 step DETR detection:
    First detect and filter with high threshold.
    Then crop frame around all detections for better res. Detect again with lower threshold.
    """

    def __init__(self, field_mask_path):
        mask_points = read_mask(field_mask_path)
        self.field_mask = create_mask(mask_points).astype(np.float32)
        # Is a measure of closeness to border. -1 outside, 1 center, 0 on border.
        self.blurred_mask = cv2.blur(self.field_mask, (FIELD_MASK_BLUR, FIELD_MASK_BLUR))
        self.blurred_mask = 2 * self.blurred_mask - 1
        # Scale to account for far people being small. 1 near, 3 far.
        self.persp_scale = create_persp_scale(mask_points)

        # Last box in detr two step; in case none found in current frame.
        self.last_detr_box = np.array([0, 0, RES[0], RES[1]], dtype=np.float32)

    def update(self, frames):
        """
        frame: ndarray (T, H, W, 3)
        motion_out: Output of Motion.update
        return: {
            boxes: All detected coarse bounding boxes.
                ndarray float (N, 4) xyxy
            player_boxes: Fine boxes of active players.
            speeds: Speeds corresponding to player_boxes.
                ndarray float (M,)
            crop: xyxy crop used for second pass.
        }
        """
        boxes_coarse, _, _, players_fine, crop_box = self.run_detr_twopass(frames[-1])
        optical_flow = run_optical_flow(frames)
        #vis_of(optical_flow)

        # Speeds corresponding to players_fine.
        speeds = []
        for box in players_fine:
            vel = compute_velocity(box, optical_flow)
            speed = np.linalg.norm(vel)
            speeds.append(speed)
        speeds = np.array(speeds, dtype=np.float32)

        return {
            "boxes": boxes_coarse,
            "player_boxes": players_fine,
            "speeds": speeds,
            "crop": crop_box,
        }

    def run_detr_twopass(self, frame):
        """
        Two pass detection. See Detector docs.
        """
        # First pass. Low person thres, high field mask thres.
        boxes_coarse = run_detr_single(frame, 0.2).astype(int)
        players_coarse = self.filter_boxes(boxes_coarse, 0.7)

        # Find bbox.
        box = extract_box(players_coarse, 150)
        if box is None:
            box = self.last_detr_box
        else:
            box = resize_bbox(box)
            self.last_detr_box = box

        # Crop and second pass. High person thres, medium field mask thres.
        x1, y1, x2, y2 = box.astype(int)
        frame_crop = frame[y1:y2, x1:x2]
        boxes_fine = run_detr_single(frame_crop, 0.4).astype(int)
        # Correct coords.
        if len(boxes_fine) > 0:
            boxes_fine[:, [0, 2]] += x1
            boxes_fine[:, [1, 3]] += y1
        players_fine = self.filter_boxes(boxes_fine, 0.5)

        return boxes_coarse, players_coarse, boxes_fine, players_fine, box

    def filter_boxes(self, boxes, thres):
        """
        Returns list of boxes that are active players.
        """
        ret = []
        for box in boxes:
            x1, y1, x2, y2 = box
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # TODO static thres for now
            if self.blurred_mask[y2, mid_x] * self.persp_scale[y2, mid_x] > thres:
                ret.append(box)

        ret = np.array(ret, dtype=np.float32)
        return ret


def run_detr_single(frame, thres):
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
        threshold=thres,
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


def run_optical_flow(frames):
    """
    frames: ndarray (T, H, W, 3)
    return: ndarray (H, W, 2) optical flow of last frame.
    """
    prev_frame = cv2.resize(frames[0], OF_RES)
    flow = np.zeros((OF_RES[1], OF_RES[0], 2), dtype=np.float32)
    for i in range(1, len(frames)):
        frame = cv2.resize(frames[i], OF_RES)
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            flow,
            0.5, 3, 15, 3, 5, 1.2, 0,
        )
        prev_frame = frame

    return flow


def compute_velocity(box, flow, persp_scale):
    """
    Compute velocity by averaging flow in middle 50% of box.
    box: xyxy, in coordinates of RES.
    flow: (H, W, 2) in coordinates of OF_RES.
    persp_scale: In coordinates of RES.
    """
    # Scale coords to account for res.
    x1, y1, x2, y2 = box
    x1 = x1 * OF_RES[0] / RES[0]
    x2 = x2 * OF_RES[0] / RES[0]
    y1 = y1 * OF_RES[1] / RES[1]
    y2 = y2 * OF_RES[1] / RES[1]

    # Average vel in middle of box.
    mid_x1 = int(0.75*x1 + 0.25*x2)
    mid_y1 = int(0.75*y1 + 0.25*y2)
    mid_x2 = int(0.25*x1 + 0.75*x2)
    mid_y2 = int(0.25*y1 + 0.75*y2)
    vel = flow[mid_y1:mid_y2, mid_x1:mid_x2].mean(axis=(0, 1))

    # Persp scale.
    scale = persp_scale[int(box[3]), int((box[0] + box[2]) / 2)]
    vel = vel * scale
    return vel


def vis_detector(frame, detector_out):
    """
    frame: cv2 format original frame.
    detector_out: Dict output of Detector.update
    """
    frame = frame.copy()

    # Draw bboxes.
    """
    for x1, y1, x2, y2 in detector_out["boxes"].astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    """
    for x1, y1, x2, y2 in detector_out["player_boxes"].astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw crop box.
    x1, y1, x2, y2 = detector_out["crop"].astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Overlay field mask
    """
    mask = detector_out["blurred_mask"] / 2 + 0.5
    mask = (mask * 255).astype(np.uint8)
    frame = cv2.addWeighted(frame, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    """

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)


def vis_of(of):
    of = torch.from_numpy(of).permute(2, 0, 1)  # (2, H, W)
    vis = flow_to_image(of).permute(1, 2, 0).numpy()  # (H, W, 3)
    cv2.imshow("Optical Flow", vis)

    cv2.waitKey(1)
