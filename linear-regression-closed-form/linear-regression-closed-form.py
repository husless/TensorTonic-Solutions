import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X1 = np.asarray(X)
    y1 = np.asarray(y)
    xx_inv = np.linalg.inv(X1.T @ X1)
    w = xx_inv @ (X1.T @ y1)
    return w