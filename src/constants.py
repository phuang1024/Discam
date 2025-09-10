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
# weight = tanh(diff / temp)
EDGE_WEIGHT_TEMP = 50

## Training parameters.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-4

# Number of simulations per epoch.
SIMS_PER_EPOCH = 10
# Number of training steps per epoch.
STEPS_PER_EPOCH = 300
# Use data from the last N epochs for training.
DATA_HISTORY = 10
# Number of epochs per training session.
#EPOCHS_PER_SESSION = 10

## Simulation parameters.
# Number of steps per simulation.
SIM_STEPS = 100
# Frame skip between steps in simulation.
SIM_FRAME_SKIP = 5
# Add randomness to initial agent bbox position.
#SIM_START_RANDOM = 40
