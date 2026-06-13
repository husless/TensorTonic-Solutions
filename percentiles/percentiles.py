import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    X = np.asarray(x)
    return np.percentile(X, q, method='linear')