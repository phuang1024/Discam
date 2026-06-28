"""
Global constants and configuration.

Image formats:
cv2 format:
    ndarray, [H, W, C], uint8 (0, 255), BGR
torch format:
    Tensor, [C, H, W], float32 (0.0, 1.0), RGB

boxes format:
    ndarray, [N, 4], int, xyxy
tracked boxes format:
    ndarray, [N, 5], int, (x, y, x, y, track_id)
"""

### Detector module params.
DET_RES = (1920, 1080)
"""Image resolution for detector module.
Since SAHI is used, this is not the input res of YOLO."""
DET_FPS = 1
"""FPS to run detector module at."""


### Classifier module params.
YX_SCALE = 2
"""Scale distances in Y direction by this amount wrt X."""
PERSP_SCALE = 3
"""From bottom to top of field mask, linearly increase distance scale from 1 to this."""


### Output params for Post Processing.
BOX_PADDING = 30
"""Padding btwn people and crop box, in RES coords."""
BOX_MIN_SIZE = 50
"""Min h and w in RES coords."""
BOX_EXPAND_EMA = 0.5
"""EMA factor when box is growing."""
BOX_SHRINK_EMA = 0.01
"""EMA factor when box is shrinking."""
BOX_SHRINK_MARGIN = 30
"""Margin before box starts to shrink."""
BOX_MOVING_AVG = 100
"""Moving average window."""

OUT_RES = (1280, 720)
"""Output video resolution."""


### Misc.
VERSION = "0.0.1"






# Trim params.
# Multiplied by FPS.
PLATEAU_LEN = 5
COUNT_THRES = 17
SPEED_THRES = 1
# Min, max time after stop to resume (sec).
MIN_STOP_TIME = 60
MAX_STOP_TIME = 120
# Min time after resume to stop.
MIN_PLAY_TIME = 10
# Margin (sec). Positive means include more footage.
TRIM_MARGIN = 5
