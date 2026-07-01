"""Global tensorboard logging utils.
"""

import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .constants import *

# Global tensorboard instance.
_logger = None
enabled = False


def init_logger(path):
    global _logger, enabled
    _logger = SummaryWriter(path)
    enabled = True


def add_scalar(tag, value, frame_i):
    if _logger is not None:
        _logger.add_scalar(tag, value, frame_i)

def add_image(tag, image, frame_i):
    """
    Args:
        image: ``cv2 format``.
    """
    if _logger is not None:
        image = cv2.resize(image, None, fx=LOG_IMG_RES, fy=LOG_IMG_RES)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.permute_dims(image, (2, 0, 1))
        _logger.add_image(tag, image, frame_i)
