"""
Person detecting using YOLO and DINO.

First, detect some bounding boxes using YOLO.
YOLO is not super reliable, so this will be a small subset, and possibly empty.

Then, encode image with DINO.
Take the average DINO feature across all bboxes to obtain the "person" embedding.
Do some temporal smoothing of this.

Then, take similarity of DINO features to the embedding,
and threshold.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils import *

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(DEVICE)
DINO.eval()

YOLO = YOLO("yolo26n")


class Detector:
    def __init__(self):
        self.person_embed = None

    def update(self, frame):
        """
        frame: cv2 format.
        return: {
            bboxes: (N, 4) xyxy bounding boxes.
            dino: (C, H', W') DINO features.
            sim: (H', W') float similarity map.
            mask: (H', W') bool thresholded sim.
        }
        """
        frame_torch = cv2_to_torch(frame).to(DEVICE)

        # Run NNs.
        bboxes = self.run_yolo(frame)
        dino_features = self.run_dino(frame_torch)

        # Update embedding.
        curr_embed = self.get_avg_embed(bboxes, dino_features)
        if curr_embed is not None:
            if self.person_embed is None:
                self.person_embed = curr_embed
            else:
                # EMA
                self.person_embed = (1 - EMBED_EMA) * self.person_embed + EMBED_EMA * curr_embed

        # Similarity map.
        embed = self.person_embed
        if embed is None:
            # Default: Use zero vector.
            embed = torch.zeros(dino_features.shape[0], device=DEVICE)

        sim = cos_similarity(dino_features, embed)
        mask = sim > DINO_THRES

        return {
            "bboxes": bboxes,
            "dino": dino_features,
            "sim": sim,
            "mask": mask,
        }

    def run_yolo(self, frame):
        """
        Run YOLO. Return bounding boxes of detected people.
        frame: cv2 format.
        return: 2D ndarray of xyxy bounding boxes. Shape (N, 4).
        """
        results = YOLO(frame, verbose=False)[0]
        bboxes = []
        for box in results.boxes:
            if box.cls == 0:
                bboxes.append(box.xyxy.cpu().numpy()[0])
        bboxes = np.array(bboxes)
        return bboxes

    def run_dino(self, frame):
        """
        Run DINO. Return feature map.
        frame: torch format.
        return: torch format, [C, H', W']
        """
        frame = resize_mul14(frame)
        features = DINO.get_intermediate_layers(frame.unsqueeze(0), n=1)[0].squeeze(0)

        new_w = frame.shape[2] // 14
        new_h = frame.shape[1] // 14
        feat_dim = features.shape[1]
        features = features.permute(1, 0).reshape(feat_dim, new_h, new_w)
        return features

    def get_avg_embed(self, bboxes, features):
        """
        Average embedding of centers of each box.
        bboxes: YOLO output.
        features: DINO output. Should be 14x smaller than original image.
        return: [C] tensor, or None if no detections.
        """
        if len(bboxes) == 0:
            return None

        embeds = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            feat_x = cx // 14
            feat_y = cy // 14
            embed = features[:, feat_y, feat_x]
            embeds.append(embed)

        embeds = torch.stack(embeds, dim=0)
        avg_embed = embeds.mean(dim=0)
        return avg_embed


def vis_detector(frame, detector_out):
    """
    frame: cv2 format original frame.
    detector_out: Dict output of Detector.update
    """
    frame = frame.copy()
    res = (frame.shape[1], frame.shape[0])

    # Overlay sim heatmap.
    sim = torch_to_cv2(detector_out["sim"])
    sim = cv2.applyColorMap(sim, cv2.COLORMAP_JET)
    sim = cv2.resize(sim, res)
    frame = cv2.addWeighted(frame, 0.7, sim, 0.3, 0)

    # Draw mask as squares.
    mask = detector_out["mask"]
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:
                top_left = (x*14, y*14)
                bottom_right = (x*14 + 14, y*14 + 14)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)

    # Draw bboxes.
    for box in detector_out["bboxes"]:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)
