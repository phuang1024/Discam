"""
PTZ control and NN inference.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from model import DiscamModel
from constants import DEVICE, MODEL_INPUT_RES
from constants2 import *

# DOFs: Pan+, Pan-, Tilt+, Tilt-, Zoom+, Zoom-
# raw_command_i = dot(DOF_ARRAY[i], nn_output)
DOF_ARRAY = np.array([
    [0, 0.5, 0, -0.5],
    [0, -0.5, 0, 0.5],
    [0.5, 0, -0.5, 0],
    [-0.5, 0, 0.5, 0],
    [0.25, 0.25, 0.25, 0.25],
    [-0.25, -0.25, -0.25, -0.25],
])


def compute_ptz(nn_output):
    """
    Compute PTZ commands from a queue of NN output.

    Control algorithm principles:
    - Zoom out rapidly.
    - Don't zoom in too soon.
    - Don't PTZ constantly.

    Control algorithm:
    - Separate movement into 6DOF: Positive and negative for each of PTZ.
    - Compute raw command for each DOF as `raw = sum(output_to_cmd(x))`
        where each `x` is an element in nn_output,
        and `output_to_cmd` is a function that maps the edge weights to a
            scalar magnitude regarding the current DOF.
    - Apply a subtractive threshold. I.e. `cmd = max(0, raw - threshold)`.
        The threshold depends on which DOF.
        E.g. zoom out has a low threshold, while zoom in has a high threshold.
    - Combine the positive and negatives into 3 DOFs, and apply an EMA to each.
    """
    # TODO
    nn_output = np.array(nn_output)
    raw_cmds = DOF_ARRAY @ nn_output


def ptz_control_thread(state: ThreadState):
    while state.run:
        # TODO ptz algorithm

        time.sleep(PTZ_INTERVAL)


@torch.no_grad()
def nn_inference_thread(state: ThreadState, model_path: Path):
    model = DiscamModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    while state.run:
        if len(state.frameq) == 0:
            time.sleep(0.01)
            continue

        frame = state.frameq[-1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, MODEL_INPUT_RES)
        frame = frame.astype("float32") / 255.0
        frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        output = model(frame)

        output = output.cpu().squeeze(0).numpy()
        state.nn_output.append(output)

        time.sleep(NN_INTERVAL)
