import numpy as np


def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    M, N = X.shape
    A = X.T @ X + lam*np.eye(N)
    w = np.linalg.inv(A) @ (X.T @ y)
    return w