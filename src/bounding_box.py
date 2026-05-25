"""
Calculate the bounding box, given the CV pipeline outputs.
"""

import torch

from pipeline import CVPipeline
from utils import *


class StaticBBox:
    """
    The goal is:
    Given the current frame (i.e. pipeline output)
    and any historical information,
    calculate the ideal bounding box for this frame.

    This *does not* consider bounding box smoothness over time.
    This *does* consider results from past frames.
    """

    def __init__(self):
        # In addition to returning a value, the most recent results are kept here.
        # For visualization purposes.
        # sim: Similarity map from DINO, lerped. torch format.
        # bbox: Bounding box coordinates. (x1, y1, x2, y2)
        self.output = {
            "sim": None,
            "bbox": None,
        }

        self.person_embed = torch.load("person.pt")
        self.person_embed /= self.person_embed.norm()

    def update(self, pipe: CVPipeline):
        """
        Call once per frame, as this will track historical data.
        return: x1, y1, x2, y2
        """
        dino = pipe.output["dino"][0]
        if dino is not None:
            sim = cos_similarity(dino, self.person_embed)
            sim = lerp(sim, 0.3, 1, 0, 1, clamp=True)
            self.output["sim"] = sim


def vis_static_bbox(bbox: StaticBBox, frame):
    """
    Draw and show a single frame on the latest results.
    frame: cv2 format
    """
    frame = frame.copy()

    # Make an overlay image based on CV output. HSV.
    overlay = np.zeros_like(frame)

    sim = bbox.output["sim"]
    if sim is not None:
        sim = torch_to_cv2(sim)
        sim = cv2.resize(sim, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay[..., 2] = sim

    # Composite.
    overlay = cv2.cvtColor(overlay, cv2.COLOR_HSV2BGR).astype(float)
    frame = np.clip(frame.astype(float) + overlay, 0, 255).astype(np.uint8)

    # Draw bbox.
    bbox_coords = bbox.output["bbox"]
    if bbox_coords is not None:
        x1, y1, x2, y2 = bbox_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("StaticBBox", frame)
    cv2.waitKey(1)
