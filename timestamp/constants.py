import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Length of data sample in frames.
VIDEO_LEN = 16
VIDEO_RES = (480, 270)

# Training parameters.
EPOCHS = 40
BATCH_SIZE = 4
LR = 1e-4
