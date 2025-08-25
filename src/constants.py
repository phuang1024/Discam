"""
Global constants, mainly for training.
"""

VIDEO_RES = (1920, 1080)
MODEL_INPUT_RES = (640, 360)

# Pixels per step.
AGENT_VELOCITY = 10

# Number of steps per simulation.
SIM_STEPS = 100
# Frame skip between steps in simulation.
SIM_FRAME_SKIP = 3
# Number of simulations per epoch.
SIMS_PER_EPOCH = 100
# Number of training steps per epoch.
STEPS_PER_EPOCH = 500
# Number of epochs per training session.
EPOCHS_PER_SESSION = 10
