"""
Run DINO on video, frames independent.
Take a reference embedding on frame 1 at user chosen pixel.
Display similarity map through video.
"""

import cv2
import numpy as np
import torch
from tqdm import tqdm

VIDEO = "../data/videos/test/Irwin_5s10f.mp4"

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
dino.eval()

#featup = torch.hub.load("mhamilton723/FeatUp", "dinov2_vitb14_featup")
#featup.eval()


def draw_heatmap(frame, sim_map):
    """
    frame: H W 3 uint8 BGR
    sim: H' W' float32
    """
    width = frame.shape[1]
    height = frame.shape[0]

    sim_map_norm = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
    heatmap = cv2.applyColorMap((sim_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (width, height))
    # Overlay heatmap on original frame
    overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    return overlay


def main():
    video_in = cv2.VideoCapture(VIDEO)
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter(
        "dino_test.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_in.get(cv2.CAP_PROP_FPS),
        (width, height),
    )

    ref_pixel = (366, 300)
    ref_embed = None

    total_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames)
    while True:
        ret, frame = video_in.read()
        if not ret:
            break

        # To tensor [0, 1] CHW
        img_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Run DINO. features: 1 C H' W'
        with torch.no_grad():
            features = dino.get_intermediate_layers(img_tensor, n=1, reshape=True)[0]
            #features = upsampler(features)

        if ref_embed is None:
            patch_x = ref_pixel[0] // 14
            patch_y = ref_pixel[1] // 14
            ref_embed = features[0, :, patch_y, patch_x]

        # Compute similarity map
        sim_map = torch.nn.functional.cosine_similarity(features.squeeze(0), ref_embed.unsqueeze(-1).unsqueeze(-1), dim=0)
        sim_map = sim_map.cpu().numpy()

        overlay = draw_heatmap(frame, sim_map)
        video_out.write(overlay)

        pbar.update(1)

    video_in.release()
    video_out.release()


if __name__ == "__main__":
    main()
