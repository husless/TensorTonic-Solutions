import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    max_x = np.max(x, axis=-1, keepdims=True)
    z = np.exp(np.asarray(x) - max_x)
    return z / np.sum(z, axis=-1, keepdims=True)
    