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
Edge weights is a command to shift the bbox.
    In deployment, the NN will continuously compute edge weights,
    which are converted into PTZ movements.
    In training, a ground truth edge weight is calculated
    from the difference between the agent bbox and the ground truth bbox.
"""

import torch

## Video parameters.
VIDEO_RES = (1920, 1080)
MODEL_INPUT_RES = (640, 360)

## Agent parameters.
# Pixels per step.
AGENT_VELOCITY = 8
# Temperature for model output.
EDGE_WEIGHT_TEMP = 50

## Training parameters.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-4

# Number of simulations per epoch.
SIMS_PER_EPOCH = 5
# Number of training steps per epoch.
STEPS_PER_EPOCH = 300
# Use data from the last N epochs for training.
DATA_HISTORY = 10
# Frequency of applying augmentations.
AUG_FREQ = 0.2

## Simulation parameters.
# Number of steps per simulation.
SIM_STEPS = 200
# Frame skip between steps in simulation.
SIM_FRAME_SKIP = 5
# Frequency of random perturbations during simulation.
SIM_RAND_FREQ = 0.05
# Magnitude of random perturbations. Applied as edge weights for a single step.
# So the expected movement is SIM_RAND_MAG * AGENT_VELOCITY.
SIM_RAND_MAG = 10
