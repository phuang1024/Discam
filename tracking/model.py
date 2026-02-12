"""
NN for classifying trajectories.
"""

import torch
import torch.nn as nn

from .constants import *


class TrackClassifier(nn.Module):
    """
    (B, 5, N) -> (B, 2)
    Tracks and mask to classification.
    Tracks input is (batch, time, (mask, x, y, vel_x, vel_y)).
        mask is 1 or 0. 0 if it's a padding index, 1 otherwise.
    Output is class probabilities.
        Index 0 is "active". Index 1 is "inactive".
    """

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(5, 8, 5, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.AvgPool1d(2),

            nn.Conv1d(8, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.AvgPool1d(2),

            nn.Conv1d(16, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )

        in_features = TRACK_LEN // 4 * 16
        self.head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x
