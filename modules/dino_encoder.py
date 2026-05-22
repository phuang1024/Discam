"""
Encode frames using DINO.
"""

import argparse

import cv2
import torch

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


if __name__ == "__main__":
    from vis import *

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    img = read_frame(args.input)

    img = resize_mult14(img)
    features = encode_image(img, 1)
    print("Features shape:", features.shape)
    print("Features dtype:", features.dtype)
    print("Features min/max:", torch.min(features), torch.max(features))

    features = features[0]
    pca_vis = visualize_pca(features)
    cv2.imshow("a", pca_vis)
    cv2.waitKey(0)
