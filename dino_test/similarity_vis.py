"""
Interactive DINO similarity visualization.
"""

import cv2
import numpy as np
import torch

from detect_people import draw_heatmap

torch.set_grad_enabled(False)

IMAGE = "../data/videos/test/Irwin_5s10f.jpg"

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
dino.eval()

ref_embed = None

# Run DINO once on image.
frame = cv2.imread(IMAGE)
img_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).float() / 255.0
with torch.no_grad():
    features = dino.get_intermediate_layers(img_tensor, n=1, reshape=True)[0]


def mouse_callback(event, x, y, flags, param):
    """
    Sets ref_embed to the clicked patch.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        global ref_embed
        patch_x = x // 14
        patch_y = y // 14
        ref_embed = features[0, :, patch_y, patch_x]
        print(f"Selected pixel: {x}, {y}")


def main():
    cv2.namedWindow("a")
    cv2.setMouseCallback("a", mouse_callback)

    while True:
        if ref_embed is None:
            cv2.imshow("a", frame)
        else:
            # Compute similarity map
            sim_map = torch.nn.functional.cosine_similarity(features.squeeze(0), ref_embed.unsqueeze(-1).unsqueeze(-1), dim=0)
            sim_map = sim_map.cpu().numpy()
            overlay = draw_heatmap(frame, sim_map)
            cv2.imshow("a", overlay)

        if cv2.waitKey(1000) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
