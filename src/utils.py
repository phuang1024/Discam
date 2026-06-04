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
FPS = 8

# VidStab window.
STAB_WINDOW = 30
# Bottom edge is this factor of original size. 1 means no warp.
WARP_CORRECTION = 0.5

# Detector params.
# Run every N frames.
DETECT_INTERVAL = 5
# Field mask edges blur scale.
FIELD_MASK_BLUR = 2
# Spectator occupancy map increase / decrease factors.
OCCU_INC_FAC = 0.05
OCCU_DEC_FAC = 0.02
# Threshold to be considered spectator.
SPECTATOR_THRES = 0.2

# Optical flow params.
# TODO for scale, at 8fps, max OF is around 5 to 10 magnitude
# Size of salience patch; i.e. downscale factor.
OF_PATCH_SIZE = 14
# Threshold to be considered fast.
OF_FAST_THRES = 2
OF_FAST_SCALE = 0.5
# Passive exponential decay on every iter.
OF_DECAY_FAC = 0.05
# Speed scaling when applying OF.
OF_APPLY_SPEED = 1

# Output params.
OUT_RES = (1280, 720)
OUT_ASPECT = 16 / 9
# This is in coordinates of RES.
BBOX_MIN_SIZE = 50


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
