"""
Misc utils.

Image formats:
cv2 format:
    ndarray, [H, W, C], uint8 (0, 255), BGR
torch format:
    Tensor, [C, H, W], float32 (0.0, 1.0), RGB
"""

import cv2
import numpy as np
import torch


def cv2_to_torch(img):
    """
    Convert image format cv2 -> torch
    """
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
    img = img.permute(1, 2, 0)
    img = (img * 255).clamp(0, 255).byte()
    img = img.cpu().numpy()
    if img.shape[2] == 3:
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


def vis_pca3(img):
    """
    3 axis PCA.
    img: torch format, [C, H, W]
    return: torch format, [3, H, W]
    """
    c, h, w = img.shape

    img_flat = img.view(c, -1).T
    img_flat = img_flat - img_flat.mean(dim=0)
    # Sample 1000 pixels for performance.
    indices = torch.linspace(0, img_flat.shape[0] - 1, 1000, dtype=torch.long)
    sample = img_flat[indices]

    cov = sample.T @ sample
    eigvals, eigvecs = torch.linalg.eig(cov)
    idx = torch.argsort(eigvals.real, descending=True)[:3]
    eigvecs = eigvecs[:, idx].real

    pca_img = (img_flat @ eigvecs).T.view(3, h, w)
    return pca_img


def vis_optical_flow(img):
    """
    Visualize optical flow.
    Hue is direction, value is magnitude.
    img: torch format, [2, H, W]
    return: cv2 format
    """
    flow = img.cpu().numpy()
    mag, ang = cv2.cartToPolar(flow[0], flow[1])
    mag = np.sqrt(mag)

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb
