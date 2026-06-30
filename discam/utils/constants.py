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
DET_THRES = 0.1
"""YOLO detection threshold."""


### Perspective module params.
PERSP_INTERVAL = -1
"""Recompute every N iters. Set to -1 to compute once at beginning."""
DEPTH_SAMPLING = 12
"""Sample depth map at pixel intervals of this size.
Should not be small, as number of samples increases quadratically."""
DEPTH_YLIMIT = 0.4
"""Sample a fraction of the image's Y extent, starting from the bottom.
TODO this is a hack to extract the linear region before the vanishing point."""

# Trig and geometry parameters.
CAM_HEIGHT = 4
"""Cam height in meters."""
CAM_FOV = 74
"""Cam horizontal FOV at widest setting."""
PERSP_MIN_SIZE = 0.01
"""Min size factor clamp (when boxes are close to vanishing point)."""


### Classifier module params.
POS_THRES = 4
"""Field mask thres for initial classification."""
MAYBE_POS_THRES = 1
"""Thres for consideration during filtering."""
ACTIVE_STD_THRES = 3
"""Std Z score threshold for stddev filtering."""

SEP_EPS = 5
"""Add this value to cov eigenvalues."""


### Output params for Post Processing.
BOX_PADDING = 50
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
OUT_FPS_DOWNSCALE = 1
"""out_fps = in_fps / scale"""


### Post Processing video trim params.
TRIM_MED_FILTER = 7
"""Trim median filter size."""
TRIM_PLATEAU = 10
"""Signals need to exceed thres for this many (CV) iters."""
TRIM_COUNT_HIGH = 18
"""Active person count high thres to detect point end."""
TRIM_SEP_LOW = 1.5
TRIM_SEP_HIGH = 2.5
"""Sep metric must go above high and below low to detect point start."""

TRIM_MIN_STOP = 60
TRIM_MAX_STOP = 120
"""Min, max time after stop to resume (sec)."""
TRIM_MIN_PLAY = 10
"""Min time after resume to stop."""
TRIM_MARGIN = 5
"""Margin (sec). Positive means include more footage."""


### Tensorboard logging params.
LOG_IMG_INTERVAL = 20
"""Log image every N CV pipeline iterations."""
LOG_IMG_RES = 0.5
"""Log image resolution factor."""


### Misc.
VERSION = "0.1.0"
