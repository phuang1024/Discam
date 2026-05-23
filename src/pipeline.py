"""
Pipeline class and related utilities.
"""

import cv2
import torch

from constants import *
from utils import *

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
        # dino: Tensor, [N, C, H', W']
        # of: Tensor, [2, H, W], (dx, dy)
        # bgr: torch format image
        self.output = {
            "dino": None,
            "of": None,
            "bgr": None,
        }

        self.frame_i = 0

    def update(self, frame) -> None:
        """
        Run each CV component. Stores results internally.

        frame: cv2 format.
        """
        print("Pipeline update frame", self.frame_i)

        frame_cv2 = frame
        frame_torch = cv2_to_torch(frame_cv2).to(DEVICE)

        if self.frame_i % DINO_INTERVAL == 0:
            self.output["dino"] = run_dino(frame_torch)

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
    print(features.shape)
    return features


def vis_pipeline(pipeline: Pipeline):
    """
    Show the latest outputs.
    """
    out = pipeline.output

    if out["dino"] is not None:
        print("DINO output shape:", out["dino"].shape)
        pca_img = pca_3axis(out["dino"][0])
        pca_img = torch_to_cv2(pca_img)
        # Scale up 14x
        pca_img = cv2.resize(pca_img, None, fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("DINO PCA", pca_img)

    cv2.waitKey(1)
