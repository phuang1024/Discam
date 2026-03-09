import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frame step when creating data from raw video.
FRAME_STEP = 10
# Length of data sample in frames.
VIDEO_LEN = 16
# Resolution of data sample.
VIDEO_RES = (480, 270)

# Training parameters.
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-4
