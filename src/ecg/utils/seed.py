
"""Utilities for reproducible machine-learning experiments."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
