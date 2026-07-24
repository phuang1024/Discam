"""Camera pose estimation utils.
"""

from ..utils.constants import *

# Center of camera view pixel coords.
CENTER = (CV_RES[0] / 2, CV_RES[1] / 2)
# Pixels per degree of default camera view.
PX_PER_DEG = CV_RES[0] / CAM_FOV


def apply_ptz(points, ptz):
    """Convert points from original camera view to given PTZ pos.

    Given pixel positions of points in original ``p,t,z = 0`` camera view coordinates.
    Compute the pixel positions of these points at a given PTZ pos.

    Args:
        points: ``ndarray float (N, 2)``, 2D pixel positions.
        ptz: ``ptz format`` camera pos.

    Returns:
        Same format as ``points``.
    """
    points = points.copy()
    # Apply pan tilt.
    points[:, 0] -= PX_PER_DEG * ptz[0]
    points[:, 1] -= PX_PER_DEG * ptz[1]

    # Apply zoom.
    delta = points - CENTER
    points = CENTER + delta * ptz[2]
    return points


def apply_inverse_ptz(points, ptz):
    """Inverse of ``apply_ptz``.
    """
    # Apply inverse zoom.
    delta = points - CENTER
    points = CENTER + delta / ptz[2]

    # Apply inverse pan tilt.
    points[:, 0] += PX_PER_DEG * ptz[0]
    points[:, 1] += PX_PER_DEG * ptz[1]
    return points
