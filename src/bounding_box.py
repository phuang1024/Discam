"""
Calculate the bounding box, given the CV pipeline outputs.
"""

import torch

from pipeline import CVPipeline
from constants import *
from utils import *


class StaticBBox:
    """
    The goal is:
    Given the current frame (i.e. pipeline output)
    and any historical information,
    calculate the ideal bounding box for this frame.

    This *does not* consider bounding box smoothness over time.
    This *does* consider results from past frames.

    DINO: Static thresholding.
    OF, BGR: Dynamic threshold:
        Images are normalized to 0-1 every frame, and compared to a fixed thres.
        Therefore, when there are a small amount of clear tracks, those will be highlighted.
        When not many strong tracks, much of the space will be highlighted.
    """

    def __init__(self):
        # In addition to returning a value, the most recent results are kept here.
        # For visualization purposes.
        # bbox: Bounding box coordinates. (x1, y1, x2, y2)
        # dino: Thresholded DINO. torch format
        # motion: OF and BGR multiplied. torch format
        self.output = {
            "bbox": None,
            "dino": None,
            "motion": None,
        }

        self.person_embed = torch.load("person.pt")
        self.person_embed /= self.person_embed.norm()

    def dynamic_thres(self, img, thres):
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        return (img > thres).float()

    def update(self, pipe: CVPipeline):
        """
        Call once per frame, as this will track historical data.
        return: x1, y1, x2, y2
        """
        dino = pipe.output["dino"][0]
        if dino is not None:
            sim = cos_similarity(dino, self.person_embed)
            # Blur
            sim = sim.unsqueeze(0).unsqueeze(0)
            sim = torch.nn.functional.avg_pool2d(sim, kernel_size=3, stride=1, padding=1)
            sim = sim.squeeze(0).squeeze(0)

            sim = (sim > DINO_THRES).float()
            self.output["dino"] = sim

        of = pipe.output["of"]
        bgr = pipe.output["bgr"]
        if of is not None and bgr is not None:
            of = torch.norm(of, dim=0)
            of = self.dynamic_thres(of, OF_THRES)
            bgr = bgr[0]
            bgr = self.dynamic_thres(bgr, BGR_THRES)

            #motion = of * bgr
            motion = of
            self.output["motion"] = motion


def vis_static_bbox(bbox: StaticBBox, frame):
    """
    Draw and show a single frame on the latest results.
    frame: cv2 format
    """
    frame = frame.copy()

    # Make an overlay image based on CV output. HSV.
    overlay = np.zeros_like(frame)

    sim = bbox.output["dino"]
    if sim is not None:
        sim = torch_to_cv2(sim)
        sim = cv2.resize(sim, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay[sim > 0, 2] = 64

    motion = bbox.output["motion"]
    if motion is not None:
        motion = torch_to_cv2(motion)[..., 0]
        overlay[motion > 0, 1] = 255

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
