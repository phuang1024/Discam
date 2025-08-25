"""
Global constants, mainly for training.
"""

import torch

VIDEO_RES = (1920, 1080)
MODEL_INPUT_RES = (640, 360)

# Pixels per step.
AGENT_VELOCITY = 10
# weight = tanh(diff / temp)
EDGE_WEIGHT_TEMP = 40

# Number of steps per simulation.
SIM_STEPS = 100
# Frame skip between steps in simulation.
SIM_FRAME_SKIP = 3
# Number of simulations per epoch.
SIMS_PER_EPOCH = 1#00
# Number of training steps per epoch.
STEPS_PER_EPOCH = 500
# Number of epochs per training session.
EPOCHS_PER_SESSION = 10

BATCH_SIZE = 16
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
