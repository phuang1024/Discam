import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters.
VIDEO_LEN = 64
# Around the "point start" event, this many frames labeled as 1.
POS_LABEL_RADIUS = 3

# Training parameters.
EPOCHS = 40
BATCH_SIZE = 4
LR = 1e-4
