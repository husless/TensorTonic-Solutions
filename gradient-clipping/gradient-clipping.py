import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    v = np.asarray(g)
    if max_norm <= 0.0:
        return v

    norm = np.linalg.norm(v)
    if norm < 1.0E-9:
        return v

    return v * min(1.0, max_norm/norm)