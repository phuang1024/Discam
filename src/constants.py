import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# See dataset
TRANSL_FAC = 200
SCALE_FAC = 400
