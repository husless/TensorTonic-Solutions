import numpy as np


def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v1 = np.asarray(v)
    w1 = np.asarray(w)
    deno = np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(w1,w1))
    if np.allclose(deno, 0, 1.0E-8):
        return np.nan
    
    return np.arccos(np.clip(np.dot(v1,w1)/deno, -1, 1))