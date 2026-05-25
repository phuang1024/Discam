import torch

# Input video res/fps.
RES = (960, 540)
FPS = 8

# VidStab window.
STAB_WINDOW = 30
# Run DINO every N frames in Pipeline.
DINO_INTERVAL = 5
# Bottom edge is this factor of original size. 1 means no warp.
WARP_CORRECTION = 0.5

DINO_THRES = 0.2
OF_THRES = 0.3
BGR_THRES = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
