"""Deterministic reproducibility helpers across Python, NumPy, and PyTorch."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set seeds for deterministic behavior across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
