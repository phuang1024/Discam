import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters.
VIDEO_LEN = 128
# Around the "point start" event, this many frames labeled as 1.
POS_LABEL_RADIUS = 30

# Training parameters.
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4
POS_WEIGHT = 10
