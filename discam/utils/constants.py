"""
Global constants and configuration.

Image formats:
cv2 format:
    ndarray, [H, W, C], uint8 (0, 255), BGR
torch format:
    Tensor, [C, H, W], float32 (0.0, 1.0), RGB

boxes format:
    ndarray, [N, 4], int, xyxy
"""

### Detector module params.
DET_RES = (1920, 1080)
"""Image resolution for detector module.
Since SAHI is used, this is not the input res of YOLO."""
DET_FPS = 1
"""FPS to run detector module at."""


### Perspective module params.
PERSP_QSIZE = 20
"""Number of past frames to consider."""
PERSP_EMA = 0.05
"""EMA fac of linear model params over time."""
CAM_HEIGHT = 4
"""Cam height in meters."""
CAM_FOV = 74
"""Cam horizontal FOV at widest setting."""
PERSP_MIN_SIZE = 0.01
"""Min size factor clamp (when boxes are close to vanishing point)."""


### Classifier module params.
POS_THRES = 4
"""Field mask thres for initial classification."""
MAYBE_POS_THRES = 0
"""Thres for consideration during filtering."""
ACTIVE_STD_THRES = 2.5
"""Std Z score threshold for stddev filtering."""


### Output params for Post Processing.
BOX_PADDING = 50
"""Padding btwn people and crop box, in RES coords."""
BOX_MIN_SIZE = 50
"""Min h and w in RES coords."""
#BOX_MEDIAN_FILTER = 3
"""Median filter size."""
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
VERSION = "0.1.0"





"""
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
"""
