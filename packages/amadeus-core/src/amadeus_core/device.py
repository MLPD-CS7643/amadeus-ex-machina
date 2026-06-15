"""
Device selection and seeding.

Single source of truth for "what torch device should I use" (previously duplicated
between ``utils/model_utils.get_device`` and ``app/inference.select_device``).
"""

import random

import numpy as np
import torch

RANDOM_SEED = 42


def get_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Backwards-compatible alias used by the old inference engine.
select_device = get_device


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed torch / numpy / random for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
