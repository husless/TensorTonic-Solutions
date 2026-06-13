import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    x = np.asarray(v)
    if x.ndim == 1:
        return np.sqrt(np.dot(x,x))

    return np.sqrt(np.sum(x**2, axis=1))