import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    A = np.asarray(X)
    if A.ndim != 2:
        return None
        
    N, _ = A.shape
    if N < 2:
        return None

    Ac = A - np.mean(A, axis=0)
    sigma = (Ac.T @ Ac) / (N - 1)
    return sigma