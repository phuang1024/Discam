"""
This file defines the Agent class.

The agent uses the NN to control a virtual PTZ camera (a bounding box).

This is used for training and testing, simulating a PTZ camera on a fixed frame video.
"""

from collections import deque

from constants import *


class Agent:
    """
    Virtual PTZ camera agent.

    Agent bounding box is self.bbox.

    TODO data type

    self.bbox coordinates correspond to the original video resolution.
    This is whatever video is used for training or testing.
    This means self.bbox does not necessarily correspond to pixel indices of the model's input,
    because model input res is defined by MODEL_INPUT_RES.
    """

    def __init__(self, model, video_res, velocity):
        """
        video_res: (width, height) of input video frames.
        """
        self.model = model
        self.video_res = video_res
        self.velocity = velocity

        self.bbox = (0, 0, video_res[0] / 2, video_res[1] / 2)

        # Frames for RNN input. Most recent is index 0.
        buffer_len = max(1, (RNN_FRAMES - 1) * RNN_STEP + 1)
        self.frames = deque(maxlen=buffer_len)

    def clear_buffer(self):
        self.frames.clear()

    def step(self, frame):
        """
        frame: (3, H, W) RGB float32 tensor [0, 1]
            Make sure this frame is the correct resolution, and crop (if applicable; during training).
            Frame is given as is to model.
        return: Predicted edge weights.
            ndarray float32 (4,)
        """
        frame = frame.unsqueeze(0)  # (1, 3, H, W)
        self.frames.appendleft(frame)

        rnn_frames = []
        for i in range(RNN_FRAMES):
            idx = i * RNN_STEP
            if idx < len(self.frames):
                rnn_frames.append(self.frames[idx])
            else:
                rnn_frames.append(self.frames[-1])

        x = torch.cat(rnn_frames, dim=1)  # (1, 3*N, H, W)
        pred = self.model(x)[0]
        pred = pred.detach().cpu().numpy()

        aspect = self.video_res[0] / self.video_res[1]
        bbox = apply_edge_weights(self.bbox, pred, aspect, self.velocity)
        self.set_bbox(bbox)

        return pred

    def set_bbox(self, bbox):
        """
        Checks bounds after setting.

        bbox: (x1, y1, x2, y2)
        """
        self.bbox = bbox
        self.check_bbox_bounds()

    def check_bbox_bounds(self):
        """
        Ensure bbox is within and smaller than video frame.
        Maintains original bbox size (unless bbox is larger than frame).
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
    return: New (x1, y1, x2, y2)
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
