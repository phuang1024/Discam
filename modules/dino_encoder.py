"""
Encode frames using DINO.
"""

import argparse

import cv2
import numpy as np
import torch

from video_read import ScaledReader

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
DINO.eval()


def resize_mult14(img):
    """
    Resize image so width and height are a multiple of 14.
    Rounds down to the nearest mult of 14.
    Required to use DINO.

    img: cv2 format.
    """
    new_width = img.shape[1] // 14 * 14
    new_height = img.shape[0] // 14 * 14
    return cv2.resize(img, (new_width, new_height))


@torch.no_grad()
def encode_image(img, num_hidden):
    """
    img: cv2 format.
    return: [N, C, H, W], tensor, float32, [-inf, inf].
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img.float() / 255.0

    features = DINO.get_intermediate_layers(img, n=num_hidden, reshape=True)[0]
    return features


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    cap = ScaledReader(args.input)
    ret, img = cap.read()
    assert ret

    img = resize_mult14(img)
    features = encode_image(img, 1)
    print("Features shape:", features.shape)
    print("Features dtype:", features.dtype)
    print("Features min/max:", torch.min(features), torch.max(features))

    features = features[0]
    pca_vis = visualize_pca(features)
    # Resize up
    pca_vis = cv2.resize(pca_vis, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("a", pca_vis)
    cv2.waitKey(0)
