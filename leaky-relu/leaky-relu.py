import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    z = np.asarray(x, dtype=float)
    cond = (z < 0)
    z[cond] *= alpha
    return z
