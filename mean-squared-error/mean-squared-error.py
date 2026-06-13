import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    res = np.asarray(y_pred) - np.asarray(y_true)
    return np.mean(res**2)
