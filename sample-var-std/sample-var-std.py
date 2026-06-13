import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    std = np.std(x, ddof=1)
    return std**2, std