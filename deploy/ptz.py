"""
PTZ control and NN inference.
"""

import time
from pathlib import Path

import cv2
import torch

from model import DiscamModel
from constants import DEVICE, MODEL_INPUT_RES
from constants2 import *


def ptz_control_thread(state: ThreadState):
    pass


def nn_inference_thread(state: ThreadState, model_path: Path):
    model = DiscamModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

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
        with torch.no_grad():
            output = model(frame)

        output = output.cpu().squeeze(0).numpy()
        state.nn_output.append(output)

        print(state.nn_output)
        time.sleep(NN_INTERVAL)
