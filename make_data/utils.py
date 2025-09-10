"""
Utilities.
"""


class EMA:
    """
    Temporal exponential moving average.
    """

    def __init__(self, alpha=0.8):
        """
        alpha: Smoothing factor, between 0 and 1.
        """
        self.alpha = alpha
        self.ema = None

    def update(self, x):
        """
        x: New input, numpy array.
        """
        if self.ema is None:
            self.ema = x
        else:
            self.ema = (self.ema * self.alpha) + (x * (1 - self.alpha))
        return self.ema


def bbox_aspect(bbox, aspect, width, height):
    """
    Adjust bounding box to given aspect ratio.
    Will always expand box.
    Respects image boundaries.

    Warning: Does nothing if bbox ends up larger than image.

    bbox: (x1, y1, x2, y2)
    aspect: width / height
    """
    x1, y1, x2, y2 = bbox
    curr_aspect = (x2 - x1) / (y2 - y1)

    if curr_aspect > aspect:
        # Too wide, increase height
        new_h = (x2 - x1) / aspect
        center_y = (y1 + y2) / 2
        y1 = int(max(0, center_y - new_h / 2))
        y2 = int(min(height, center_y + new_h / 2))

    elif curr_aspect < aspect:
        # Too tall, increase width
        new_w = (y2 - y1) * aspect
        center_x = (x1 + x2) / 2
        x1 = int(max(0, center_x - new_w / 2))
        x2 = int(min(width, center_x + new_w / 2))

    # Check bounds.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 > width:
        x1 -= (x2 - width)
        x2 = width
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > height:
        y1 -= (y2 - height)
        y2 = height

    return (x1, y1, x2, y2)
