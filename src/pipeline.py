"""
Pipeline class and related utilities.
"""

import cv2
import torch

from constants import *
from utils import *
from utils import cv2_to_torch

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(DEVICE)
DINO.eval()


class Pipeline:
    """
    The Pipeline handles all the computer vision inference.
    Takes in a source (e.g. video or camera feed).
    Outputs embeddings from the various components.

    Input: Sequence of frames. Call update() on each frame.
    This class will *not* resize or check fps on the given frame;
        i.e. do it before calling update().

    Components are:
        DINO; Optical Flow; Background Removal.
    OF and BG removal are run every frame.
    DINO is run every N frames.

    Maintains the latest output from each component.

    Supports tiled inference on all components.
    """

    def __init__(self):
        # original: cv2 format original image.
        # dino: Tensor, [N, C, H', W']
        # of: Tensor, [2, H, W], (dx, dy)
        # bgr: Tensor, [H, W], mask
        self.output = {
            "original": None,
            "dino": None,
            "of": None,
            "bgr": None,
        }

        self.frame_i = 0
        self.bg_remover = BgRemover()

    def update(self, frame) -> None:
        """
        Run each CV component. Stores results internally.

        frame: cv2 format.
        """
        print("Pipeline update frame", self.frame_i)

        frame_cv2 = frame
        frame_torch = cv2_to_torch(frame_cv2).to(DEVICE)

        self.output["original"] = frame_cv2

        if self.frame_i % DINO_INTERVAL == 0:
            self.output["dino"] = run_dino(frame_torch)

        self.output["bgr"] = self.bg_remover.remove_bg(frame_cv2).to(DEVICE)

        self.frame_i += 1


def run_dino(frame, num_hidden=3):
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


class BgRemover:
    """
    Wrapper around cv2 BG remover.
    """

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        # TODO params.

    def remove_bg(self, frame):
        """
        frame: cv2 format.
        return: torch format, binary mask.
        """
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        # TODO tune threshold
        fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]
        fg_mask = fg_mask[..., None]
        fg_mask = cv2_to_torch(fg_mask)
        return fg_mask


def vis_pipeline(pipeline: Pipeline):
    """
    Show the latest outputs.
    """
    out = pipeline.output

    cv2.imshow("Original", out["original"])

    if out["dino"] is not None:
        print("DINO output shape:", out["dino"].shape)
        pca_img = pca_3axis(out["dino"][0])
        pca_img = torch_to_cv2(pca_img)
        # Scale up 14x
        pca_img = cv2.resize(pca_img, None, fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("DINO PCA", pca_img)

    if out["bgr"] is not None:
        print("BG mask shape:", out["bgr"].shape)
        bgr_img = torch_to_cv2(out["bgr"])
        cv2.imshow("BG Mask", bgr_img)

    cv2.waitKey(1)
