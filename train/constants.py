"""
Global constants, mainly for training.

Note on the difference between velocity and temp:
Temp is used in model output: output = logit / temp.
    Ideally, output should be near [-1, 1], because ground truth edge weights are in this range.
    This gives the logits more dynamic range.
Output is then multiplied with velocity when simulating, to determine how many pixels to move.

Note on the difference between bbox and edge weights:
Bounding box is a region of the frame.
    The agent has a bbox (it's current view).
    Ground truth bboxes are generated based on motion detection.
    Bboxes are in (x1, y1, x2, y2) format.
Edge weights is a command to shift the bbox.
    In deployment, the NN will continuously compute edge weights,
    which are converted into PTZ movements.
    In training, a ground truth edge weight is calculated
    from the difference between the agent bbox and the ground truth bbox.
    Positive edge weight means expand edge (increase bbox area).
"""

import json

import torch

## Video parameters.
VIDEO_RES = (1920, 1080)
MODEL_INPUT_RES = (640, 360)

## Agent parameters.
# Pixels per step.
AGENT_VELOCITY = 5
# Temperature for model output.
EDGE_WEIGHT_TEMP = 100

## Training parameters.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-4

# Loss scaling factor for terms that expand bbox.
LOSS_EXPAND_FACTOR = 2

# Offset between bbox and frame.
BBOX_FRAME_OFFSET = 10

# Number of simulations per epoch.
SIMS_PER_EPOCH = 10
# Number of training steps per epoch.
STEPS_PER_EPOCH = 300
# Use data from the last N epochs for training.
DATA_HISTORY = 20
# Frequency of applying augmentations.
AUG_FREQ = 0.2

## Simulation parameters.
# Number of steps per simulation.
# So, each simulation spans SIM_STEPS * SIM_FRAME_SKIP frames of the video.
SIM_STEPS = 100
# Frame skip between steps in simulation.
SIM_FRAME_SKIP = 5
# Frequency of random perturbations during simulation.
SIM_RAND_FREQ = 0.1
# Magnitude of random perturbations. Applied as edge weights for a single step.
# So the average randomness is SIM_RAND_MAG * (AGENT_VELOCITY * SIM_FRAME_SKIP) pixels.
SIM_RAND_MAG = 30


def save_constants(path: str, extra_data: dict):
    """
    Save constants to a text file.

    extra_data: Additional data to save.
    """
    data = {}
    for k, v in globals().items():
        if k.isupper() and not k.startswith("_"):
            data[k] = str(v)

    data.update(extra_data)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
