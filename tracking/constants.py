import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Tracking parameters.
# Image resolution for detection and tracking. Mult of 32.
FRAME_RES = (960, 544)
# Run detection every Nth video frame.
DETECT_INTERVAL = 1
# Append to track every Nth detection. Regardless, tracking is still implicitly performed every detection.
TRACK_INTERVAL = 3
# Max number of points in a track. NN input length is this.
TRACK_LEN = 32
# Confidence thres for YOLO detection.
DETECT_CONF = 0.1

## Training parameters.
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
