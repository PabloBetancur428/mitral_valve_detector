"""
seed.py

Utilities for controlling randomness and ensuring reproducibility

Random seed for:
- Python random module
- Numpy
- Pytorch CPU
- Pytorch CUDA
-CuDNN deterministic
"""

import random
import numpy as np
import torch 

def set_seed(seed: int = 42):
    """
    Random seed for reproducibility

    Parameters
    ---------
    seed: int

    """

    # Python random
    random.seed(seed)

    # Numpy random generator
    np.random.seed(seed)

    # Pytorch CPU random seed
    torch.manual_seed(seed)

    # Pytorch CUDA random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    # CuDNN deterministic 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

