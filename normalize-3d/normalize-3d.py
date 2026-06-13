import numpy as np


def norm_3d(v):
    x = np.asarray(v)
    if x.ndim == 1:
        return np.sqrt(np.dot(x,x))

    return np.sqrt(np.sum(x**2, axis=1, keepdims=True))


def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    x = np.asarray(v)
    v_norm = norm_3d(x)

    return np.where(v_norm > 0, x/v_norm, x)
