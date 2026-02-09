import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FPS of the track. Raw track from YOLO is downsampled to this.
FPS = 5
# Max number of points in a track. NN input length is this.
TRACK_LEN = 20
