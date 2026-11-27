"""
One call to seed every random source the pipelines touch.

Usage:
    from seeding import seed_everything

    rng = seed_everything(1234)   # python, numpy and torch if installed
"""

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> np.random.Generator:
    """
    Seed python, numpy and torch (when installed) and return a numpy Generator.

    deterministic also asks torch for deterministic kernels, which can be slower.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)
