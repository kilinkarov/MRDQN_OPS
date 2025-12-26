"""Helper utility functions."""

import random

import numpy as np
import torch


def sigmoid(inx):
    """Numerically stable sigmoid function."""
    if inx >= 0:
        return 1.0 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))


def normalization(data):
    """Normalize data to [0, 1] range."""
    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        return np.zeros_like(data)
    normalized_data = (data - np.min(data)) / data_range
    return normalized_data


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
