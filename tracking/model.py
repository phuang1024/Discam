import torch
import torch.nn as nn


class TrackingModel(nn.Module):
    """
    (B, N, 4) -> (B, 2)
    Tracks to classification.
    """
