import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x1 = np.asarray(x)
    y1 = np.asarray(y)
    return float(np.sqrt(np.sum((x1-y1)**2)))