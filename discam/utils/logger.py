"""Global tensorboard logging utils.
"""

import os

import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .constants import *

# Global tensorboard instance.
_logger = None
_path = None
enabled = False


def init_logger(path):
    global _logger, _path, enabled
    _logger = SummaryWriter(path)
    _path = path
    enabled = True


def add_scalar(tag, value, frame_i):
    if _logger is not None:
        _logger.add_scalar(tag, value, frame_i)

def add_image(tag, image, frame_i):
    """Save image to both tensorboard and disk.

    Args:
        image: ``cv2 format``.
    """
    if enabled:
        # Resize.
        image = cv2.resize(image, None, fx=LOG_IMG_RES, fy=LOG_IMG_RES)
        # Save to disk.
        dir = os.path.join(_path, tag)
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(os.path.join(dir, f"{frame_i}.jpg"), image)

        # Save to tb.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.permute_dims(image, (2, 0, 1))
        _logger.add_image(tag, image, frame_i)
