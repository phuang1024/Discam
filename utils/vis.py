"""
Misc visualization tools.
"""

import cv2
import numpy as np
import torch


def read_frame(path):
    """
    If path is an image, return it.
    If video, read the first frame.
    """
    if path.endswith((".png", ".jpg")):
        return cv2.imread(path)
    else:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        assert ret, f"Failed to read first frame of {path}"
        cap.release()
        return frame


def visualize_pca(image):
    """
    Returns 3 axis RGB PCA visualization.

    image: [C, H, W] tensor
    return: [H, W, C] cv2 format.
    """
    # Sample a subset of vectors for performance.
    vectors = image.view(image.shape[0], -1).T
    indices = torch.randperm(vectors.shape[0])[:1000]
    sample = vectors[indices]

    # Run PCA
    sample_mean = sample.mean(dim=0)
    sample_centered = sample - sample_mean
    cov = sample_centered.T @ sample_centered / (sample_centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    top_eigvecs = eigvecs[:, -3:]

    # Project all vectors onto the top 3 eigenvectors.
    projected = (vectors - sample_mean) @ top_eigvecs
    projected = projected - projected.min(dim=0).values
    projected = projected / projected.max(dim=0).values
    projected = projected.T.view(3, image.shape[1], image.shape[2])

    # Convert to cv2 format.
    projected = projected.permute(1, 2, 0).cpu().numpy()
    projected = (projected * 255).astype(np.uint8)
    return projected
