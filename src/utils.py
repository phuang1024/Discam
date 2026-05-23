"""
Misc utils.

Image formats:
cv2 format:
    ndarray, [H, W, C], uint8 (0, 255), BGR
torch format:
    Tensor, [C, H, W], float32 (0.0, 1.0), RGB
"""

import cv2
import torch


def cv2_to_torch(img):
    """
    Convert image format cv2 -> torch
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1)
    return img


def torch_to_cv2(img):
    """
    Convert image format torch -> cv2
    """
    img = img.permute(1, 2, 0)
    img = (img * 255).clamp(0, 255).byte()
    img = img.cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def resize_mul14(img):
    """
    Resize image so that H and W are multiples of 14 (for DINO).
    img: torch format.
    """
    new_w = (img.shape[2] // 14) * 14
    new_h = (img.shape[1] // 14) * 14
    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear").squeeze(0)
    return img
