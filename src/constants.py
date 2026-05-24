import torch

# Input video res/fps.
RES = (640, 360)
FPS = 8

# Run DINO every N frames in Pipeline.
DINO_INTERVAL = 5
# Number of tiles for each of H and W.
TILE_COUNT = 3
# Bottom edge is this factor of original size. 1 means no warp.
WARP_CORRECTION = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
