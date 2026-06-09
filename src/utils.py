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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video res/fps for NN.
RES = (640, 360)
FPS = 2

# Detector params.
# Field mask edges blur size.
FIELD_MASK_BLUR = 50

# Output params.
OUT_RES = (1280, 720)
OUT_ASPECT = 16 / 9
# In coordinates of RES. Padding between outermost person and bbox.
OUT_PADDING = 50
OUT_MIN_SIZE = 50
# EMA smoothing.
OUT_EXPAND_EMA = 0.5
OUT_SHRINK_EMA = 0.02
OUT_SHRINK_MARGIN = 20
# Moving average.
OUT_MOVING_AVG = 200

VERSION = "0.0.1"


class EMA:
    def __init__(self, alpha):
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
