"""
Utilities and parameters.

Image formats:
cv2 format:
    ndarray, [H, W, C], uint8 (0, 255), BGR
torch format:
    Tensor, [C, H, W], float32 (0.0, 1.0), RGB
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input video res/fps.
RES = (640, 360)
FPS = 2

# VidStab window.
STAB_WINDOW = 30
# Bottom edge is this factor of original size. 1 means no warp.
#WARP_CORRECTION = 0.5

# Detector params.
# Run every N frames.
DETECT_INTERVAL = 1
# Field mask edges blur size.
FIELD_MASK_BLUR = 50

# Optical flow params.
# Temporal median filter window size.
OF_MEDIAN_SIZE = 5
# Magnitude scaling to account for camera perspective.
OF_PERSP_SCALE = 3

# Output params.
# In coordinates of RES. Padding between outermost person and bbox.
BBOX_PADDING = 50
OUT_RES = (1280, 720)
OUT_ASPECT = 16 / 9
# This is in coordinates of RES.
BBOX_MIN_SIZE = 50


class EMA:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def interp(x, from_min, from_max, to_min, to_max, clamp=False):
    y = (x - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
    if clamp:
        lower = np.minimum(to_min, to_max)
        upper = np.maximum(to_min, to_max)
        y = np.clip(y, lower, upper)
    return y


def clip_coords(x, y, res=RES):
    """
    Clip coordinates to be within res.
    """
    x = np.clip(x, 0, res[0] - 1)
    y = np.clip(y, 0, res[1] - 1)
    return x, y


def cv2_to_torch(img):
    """
    Convert image format cv2 -> torch
    """
    if len(img.shape) == 2:
        img = img[..., None]
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1)
    return img


def torch_to_cv2(img):
    """
    Convert image format torch -> cv2
    """
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    img = img.permute(1, 2, 0)
    img = (img * 255).clamp(0, 255).byte()
    img = img.cpu().numpy()
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
