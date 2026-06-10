import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x1 = np.asarray(x)
    y1 = np.asarray(y)
    return float(np.sum(np.abs(x1 - y1)))