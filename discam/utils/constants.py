"""
Global constants and configuration.

**Image formats:**

- ``cv2 format``: ``ndarray uint8 (H, W, C)``, (0, 255), BGR.
- ``torch format``: ``Tensor float32 (C, H, W)``, (0.0, 1.0), RGB.

**Bounding box format:**

- ``boxes format``: ``ndarray int (N, 4)``, xyxy.

Note that Post Processing crop boxes are the same format,
but ``N`` denotes the time dimension instead of a list of detections.

- ``Location`` refers to physical location on the field.
- ``Pixel position`` refers to position on image frame.

**Pan tilt zoom format:**

Three scalars ``p``, ``t``, ``z``.

- ``p``, ``t``: Real number, pan/tilt in degrees.
  ``0`` is the default centered pan.
- ``z``: Real number ``z >= 0``, ln of zoom factor.
  I.e. FOV is ``exp(z)`` times smaller in length than default zoom.
  Default zoom is ``z = 0``. Zooming by factor is additive (because of log).
"""

### Detector module params.
CV_RES = (1920, 1080)
CV_FPS = 1
"""Res and FPS for CV pipeline."""

DET_THRES = 0.1
"""YOLO detection threshold."""

TILE_SIZE_POST = 500
"""SAHI tile size in post processing."""
TILE_SIZE_LIVE = 800
"""SAHI tile size in live. Can be larger since view is zoomed in."""


### Perspective module params.
PERSP_INTERVAL = -1
"""Recompute camera params every N iters.
Set to -1 to compute once at beginning.
Live mode only computes once at the beginning regardless."""
DEPTH_SAMPLING = 12
"""Sample depth map at pixel intervals of this size.
Should not be too small, as number of samples increases quadratically."""
DEPTH_YLIMIT = 0.4
"""Sample a fraction of the image's Y extent, starting from the bottom.
TODO this is a hack to extract the linear region before the vanishing point."""

# Trig and geometry parameters.
CAM_HEIGHT = 4
"""Cam height in meters."""
CAM_FOV = 74
"""Cam horizontal FOV in degrees of input video."""
PERSP_MIN_SIZE = 0.01
"""Min size factor clamp (when boxes are close to vanishing point)."""


### Classifier module params.
DEF_POS_THRES = 4
"""Field mask thres for "definitely" positive initial pass."""
MAYBE_POS_THRES = 1
"""Thres for "maybe" positive second pass,
for consideration during additional filtering."""
ACTIVE_STD_THRES = 3
"""Z score threshold for "maybe" positive stddev filtering."""
FILTER_DEF_THRES = 15
"""Sep metric less than this to enable stddev filtering within definitely positive.
This ensures no erroneous filtering when teams are far apart.
"""

SEP_EPS = 5
"""For separation metric, add this value to cov eigenvalues.
This increases consideration on absolute (not relative) distance between two centroids."""


### Output params for Post Processing.
BOX_PADDING = 50
"""Padding btwn people and crop box, in CV_RES coords."""
BOX_MIN_SIZE = 50
"""Min h and w."""
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


### Post Processing video trim params.
TRIM_MED_FILTER = 7
"""Median filter size on relevant signals."""
TRIM_PLATEAU = 10
"""Signals need to exceed thres for this many CV iters."""
TRIM_COUNT_HIGH = 18
"""High thres of "active person count" to detect point end."""
TRIM_SEP_HIGH = 15
TRIM_SEP_LOW = 8
"""Sep metric must go above high and below low to detect point start."""

TRIM_MIN_STOP = 60
TRIM_MAX_STOP = 120
"""Min, max time after stop to resume (sec)."""
TRIM_MIN_PLAY = 10
"""Min time after resume to stop."""
TRIM_MARGIN_END = 3
TRIM_MARGIN_START = 3
"""Trim margins (sec). Positive means include more footage."""


### Live mode tracking params.
TRACK_MARGIN = 200
"""Maintain outermost person this many pixels from frame edge (in CV_RES coords)."""
TRACK_COUNT = 10
"""Begin zooming out if number of detected players below this."""
TRACK_PT = 0.3
"""Pan tilt speed multiplier. Should be on the order of 1."""
TRACK_ZOOM_IN = 0.3
TRACK_ZOOM_OUT = 0.8
"""Zoom speed multipliers."""


### Tensorboard logging params.
LOG_IMG_INTERVAL = 1#20
"""Log image every N CV pipeline iterations."""
LOG_IMG_RES = 0.5
"""Log image resolution factor."""


### Misc.
VERSION = "0.1.0"
