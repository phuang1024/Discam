import torch

# Input video res/fps.
RES = (960, 540)
FPS = 8

# Run DINO every N frames in Pipeline.
DINO_INTERVAL = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
