"""
Global constants, mainly for training.
"""

import torch

## Video parameters.
VIDEO_RES = (1920, 1080)
MODEL_INPUT_RES = (640, 360)

## Agent parameters.
# Pixels per step.
AGENT_VELOCITY = 8
# Note on the difference between velocity and temp:
# Temp is used in model output and training: Output = logit / temp.
#   Ideally, output should be near [-1, 1].
#   This gives the logits more dynamic range.
# This is then multiplied with velocity when simulating.
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
