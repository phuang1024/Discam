"""
Visualize hidden layers of the DiscamModel.
"""

import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "..", "train"))

import argparse

import matplotlib.pyplot as plt
import torch

from constants import *
from model import DiscamModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    model = DiscamModel()
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    w = 40
    h = 22
    c = 8

    weights = model.fc[0].weight.data.cpu()
    weights = weights.view(weights.size(0), c, h, w)

    # Now is (h, w)
    weights = weights.abs().mean(dim=0).mean(dim=0)

    plt.imshow(weights)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
