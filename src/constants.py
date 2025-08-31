"""
Global constants, mainly for training.
"""

import torch

VIDEO_RES = (1920, 1080)
MODEL_INPUT_RES = (640, 360)

# Pixels per step.
AGENT_VELOCITY = 8
# weight = tanh(diff / temp)
EDGE_WEIGHT_TEMP = 100

# Number of steps per simulation.
SIM_STEPS = 100
# Frame skip between steps in simulation.
SIM_FRAME_SKIP = 2
# Number of simulations per epoch.
SIMS_PER_EPOCH = 20
# Number of training steps per epoch.
STEPS_PER_EPOCH = 500
# Use data from the last N epochs for training.
DATA_HISTORY = 5
# Number of epochs per training session.
#EPOCHS_PER_SESSION = 10

BATCH_SIZE = 32
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
