import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Tracking parameters.
# Image resolution for detection and tracking. Mult of 32.
FRAME_RES = (960, 544)
# Run detection every Nth video frame.
DETECT_INTERVAL = 1
# Append to track every Nth detection. Regardless, tracking is still implicitly performed every detection.
TRACK_INTERVAL = 3
# Min speed in image widths per frame to be considered moving.
MIN_SPEED = 1e-3

# Max length of track for NN input.
TRACK_LEN = 64

## Training parameters.
EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-3
