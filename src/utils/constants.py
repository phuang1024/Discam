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


### Output params for Post Processing.
OUT_RES = (1280, 720)
"""Output video resolution."""


### Misc.
VERSION = "0.0.1"





# Bounding box params.
# In coordinates of RES. Padding between outermost person and bbox.
BOX_PADDING = 30
BOX_MIN_SIZE = 50
# Output median filter.
#BOX_MEDIAN_FILTER = 5
# EMA smoothing.
BOX_EXPAND_EMA = 0.5
BOX_SHRINK_EMA = 0.01
BOX_SHRINK_MARGIN = 30
# Moving average.
BOX_MOVING_AVG = 100

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
