"""
Reproducibility utilities.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Fix all random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enforce deterministic CUDA ops (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # older PyTorch
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
