import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image resolution for detection and tracking.
FRAME_RES = (960, 540)
# Run detection every Nth video frame. Result will depend on video FPS.
DETECT_INTERVAL = 1
# Append to track every Nth detection. Regardless, tracking is still performed every detection.
TRACK_INTERVAL = 3
# Max number of points in a track. NN input length is this.
TRACK_LEN = 20
