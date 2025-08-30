"""
Simulate PTZ camera controlled by neural network.
"""

import torch


class Agent:
    def __init__(self, model, video_res, velocity):
        """
        video_res: (width, height) of input video frames.
        """
        self.model = model
        self.video_res = video_res
        self.velocity = velocity

        self.bbox = (0, 0, video_res[0], video_res[1])

    def step(self, frame):
        """
        frame: (3, H, W) RGB float32 tensor [0, 1]
        """
        frame = frame.unsqueeze(0)  # (1, 3, H, W)

        pred = self.model(frame)[0]
        pred = pred.detach().cpu().numpy()

        aspect = self.video_res[0] / self.video_res[1]
        self.bbox = apply_edge_weights(self.bbox, pred, aspect, self.velocity)
        self.check_bbox_bounds()

    def check_bbox_bounds(self):
        """
        Ensure bbox is within and smaller than video frame.
        """
        x1, y1, x2, y2 = self.bbox

        if x2 - x1 > self.video_res[0]:
            self.bbox = (0, 0, self.video_res[0], self.video_res[1])
            return

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > self.video_res[0]:
            x1 -= (x2 - self.video_res[0])
            x2 = self.video_res[0]
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 > self.video_res[1]:
            y1 -= (y2 - self.video_res[1])
            y2 = self.video_res[1]

        self.bbox = (x1, y1, x2, y2)


def apply_edge_weights(bbox, edges, aspect, velocity, min_size=20):
    """
    Apply model prediction to bbox, keeping aspect ratio.

    Algorithm:
    First, shift edges by velocity * edge_value.
    Then, adjust bbox to keep aspect ratio.
      Keep center fixed, and adjust width or height to match aspect,
      while maintaining area.

    w * h = A
    w / h = aspect
    w = sqrt(A * aspect)
    h = A / w

    bbox: (x1, y1, x2, y2)
    edges: (up, right, down, left) in [-1, 1]
    aspect: Target aspect ratio (width / height).
    velocity: Max pixels to move per step.
    """
    x1, y1, x2, y2 = bbox
    x1 -= velocity * edges[3]
    x2 += velocity * edges[1]
    y1 -= velocity * edges[0]
    y2 += velocity * edges[2]

    # Check negativity and min area.
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if y2 < y1 + min_size:
        y2 = y1 + min_size
    if x2 < x1 + min_size:
        x2 = x1 + min_size

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    area = (x2 - x1) * (y2 - y1)

    new_w = (area * aspect) ** 0.5
    new_h = area / new_w

    x1 = center[0] - new_w / 2
    x2 = center[0] + new_w / 2
    y1 = center[1] - new_h / 2
    y2 = center[1] + new_h / 2

    return (x1, y1, x2, y2)
