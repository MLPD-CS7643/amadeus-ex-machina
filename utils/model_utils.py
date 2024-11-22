import random

import torch
import numpy as np

RANDOM_SEED = 42


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
