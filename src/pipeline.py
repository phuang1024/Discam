"""
Pipeline class and related utilities.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from vidstab.VidStab import VidStab

from constants import *
from utils import *
from utils import cv2_to_torch

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(DEVICE)
DINO.eval()

YOLO = YOLO("yolo26n")


class CVPipeline:
    """
    The Pipeline handles all the computer vision inference.
    Takes in a source (e.g. video or camera feed).
    Outputs embeddings from the various components.

    Input: Sequence of frames. User should call update() on each frame.
    This class will *not* resize or check fps on the given frame;
        i.e. do it before calling update().

    Components are:
        DINO; Optical Flow; Background Removal.
    OF and BG removal are run every frame.
    DINO is run every N frames.

    Output: Maintains the latest output from each component.

    Features:
    Does tiled inference on all components.
    Does warp correction on OF and BGR.
    """

    def __init__(self):
        # original: cv2 format original image.
        # yolo: List of [x1, y1, x2, y2] bounding boxes.
        # dino: Tensor, [N, C, H', W']
        # sim: Tensor, [H', W'], dino similarity map.
        # of: Tensor, [2, H, W], (dx, dy)
        # bgr: Tensor, [H, W], mask
        self.output = {
            "original": None,
            "yolo": None,
            "dino": None,
            "sim": None,
            "of": None,
            "bgr": None,
        }
        self.frame_i = 0

        self.stab = VidStab()

        self.of_module = OpticalFlow()
        self.bgr_module = BgRemover()

        self.init_warp()

    def init_warp(self):
        """
        Make perspective correction.
        Shrink lower edge by factor, forming a trapezoid shape.
        """
        length = int(RES[0] * WARP_CORRECTION)
        rect1 = np.array([
            [0, 0],
            [RES[0], 0],
            [RES[0], RES[1]],
            [0, RES[1]],
        ], dtype=np.float32)
        rect2 = np.array([
            [0, 0],
            [RES[0], 0],
            [RES[0] / 2 + length / 2, RES[1]],
            [RES[0] / 2 - length / 2, RES[1]],
        ], dtype=np.float32)

        self.warp_mat = cv2.getPerspectiveTransform(rect1, rect2)
        self.inv_warp_mat = cv2.getPerspectiveTransform(rect2, rect1)

    def apply_warp(self, img):
        """
        img: cv2 format, resolution RES.
        return: cv2 format, resolution RES.
        """
        warped = cv2.warpPerspective(img, self.warp_mat, RES)
        return warped

    def apply_inv_warp(self, img):
        inv_warped = cv2.warpPerspective(img, self.inv_warp_mat, RES)
        # Unsqueeze last dim if single channel.
        if len(inv_warped.shape) == 2:
            inv_warped = inv_warped[..., None]
        return inv_warped

    def update(self, frame) -> None:
        """
        Run each CV component. Stores results internally.

        frame: cv2 format.
        """
        print("Pipeline update frame", self.frame_i)

        # Stabilization
        stab_frame = self.stab.stabilize_frame(input_frame=frame, smoothing_window=STAB_WINDOW)
        if self.frame_i >= STAB_WINDOW:
            frame = stab_frame

        # Original image in torch format.
        frame_torch = cv2_to_torch(frame).to(DEVICE)
        # Warped image in torch format.
        frame_warped = self.apply_warp(frame)
        frame_warped = cv2_to_torch(frame_warped).to(DEVICE)

        self.output["original"] = frame

        if self.frame_i % DINO_INTERVAL == 0:
            boxes = run_yolo(frame)
            self.output["yolo"] = boxes

            dino = run_dino(frame_torch).to(DEVICE)
            self.output["dino"] = dino

            person_embed = extract_person_embed(boxes, dino)
            sim = cos_similarity(dino[0], person_embed)
            self.output["sim"] = sim

        ret = self.bgr_module.remove_bg(frame_warped)
        ret = cv2_to_torch(self.apply_inv_warp(torch_to_cv2(ret))).to(DEVICE)
        self.output["bgr"] = ret

        ret = self.of_module.compute_flow(frame_warped)
        ret = cv2_to_torch(self.apply_inv_warp(torch_to_cv2(ret))).to(DEVICE)
        self.output["of"] = ret

        self.frame_i += 1


def run_yolo(frame):
    """
    frame: cv2 format.
    return: list of bounding boxes of people, each box is (x1, y1, x2, y2).
    """
    results = YOLO(frame, classes=[0])[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    return boxes


def run_dino(frame, num_hidden=1):
    """
    frame: torch format.
    return: torch format, [N, C, H', W']
    """
    frame = resize_mul14(frame)
    features = DINO.get_intermediate_layers(frame.unsqueeze(0), n=num_hidden)
    # (N, H'*W', C)
    features = torch.stack([f.squeeze(0) for f in features], dim=0)
    # (N, C, H', W')
    new_w = frame.shape[2] // 14
    new_h = frame.shape[1] // 14
    features = features.permute(0, 2, 1).reshape(features.shape[0], features.shape[2], new_h, new_w)
    return features


def extract_person_embed(yolo_boxes, dino_features):
    """
    Extract person embedding.
    Take the average of the DINO feature in the center of each box.
    """
    embeds = []
    for box in yolo_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # Convert to DINO feature coordinates.
        cx = cx // 14
        cy = cy // 14
        embed = dino_features[0, :, cy, cx]
        embeds.append(embed)

    if len(embeds) == 0:
        return torch.zeros_like(dino_features[0, :, 0, 0])

    print(len(embeds), embeds[0].shape)
    person = torch.mean(torch.stack(embeds, dim=0), dim=0)
    print(person.shape)
    return person


class OpticalFlow:
    """
    Wrapper around cv2 Farneback.
    """

    def __init__(self):
        self.prev_frame = None
        self.prev_flow = None

    def compute_flow(self, frame):
        """
        frame: torch format.
        return: torch format, [2, H, W], (dx, dy)
        """
        frame = torch_to_cv2(frame)

        if self.prev_frame is None:
            self.prev_frame = frame
            return torch.zeros(2, frame.shape[0], frame.shape[1])

        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            self.prev_flow,
            0.5, 3, 15, 3, 5, 1.2, 0,
        )
        self.prev_frame = frame
        self.prev_flow = flow

        flow = torch.from_numpy(flow).permute(2, 0, 1)
        return flow


class BgRemover:
    """
    Wrapper around cv2 BG remover.
    """

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=8,
            detectShadows=False,
        )

    def remove_bg(self, frame):
        """
        frame: torch format.
        return: torch format, [1, H, W] float mask.
        """
        frame = torch_to_cv2(frame)

        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        fg_mask = fg_mask[..., None]
        fg_mask = cv2_to_torch(fg_mask)
        return fg_mask


def vis_pipeline(pipeline: Pipeline):
    """
    Show the latest outputs.
    """
    out = pipeline.output

    # Draw a image of detections: YOLO boxes and DINO similarity map.
    detect_img = out["original"].copy()

    # Overlay similarity map on original.
    sim = out["sim"]
    if sim is not None:
        sim = torch_to_cv2(sim)
        sim = cv2.resize(sim, RES, interpolation=cv2.INTER_NEAREST)
        sim_color = cv2.applyColorMap(sim, cv2.COLORMAP_JET)
        detect_img = cv2.addWeighted(detect_img, 0.5, sim_color, 0.5, 0)

    # Draw YOLO boxes
    yolo = out["yolo"]
    if yolo is not None:
        for box in yolo:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(detect_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Detections", detect_img)

    """
    cv2.imshow("Original", out["original"])

    if out["dino"] is not None:
        print("DINO output shape:", out["dino"].shape)
        #dino_img = torch_to_cv2(vis_pca3(out["dino"][0]))
        dino_img = vis_similarity(out["dino"][0], PERSON)
        # Scale up 14x
        dino_img = cv2.resize(dino_img, None, fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("DINO PCA", dino_img)

    if out["of"] is not None:
        print("Optical Flow shape:", out["of"].shape)
        of_img = vis_optical_flow(out["of"])
        cv2.imshow("Optical Flow", of_img)

    if out["bgr"] is not None:
        print("BG mask shape:", out["bgr"].shape)
        bgr_img = torch_to_cv2(out["bgr"])
        cv2.imshow("BG Mask", bgr_img)
    """

    cv2.waitKey(1)
