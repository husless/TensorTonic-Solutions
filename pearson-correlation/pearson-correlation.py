import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    A = np.asarray(X)
    if A.ndim != 2:
        return None

    N, _ = A.shape
    if N < 2:
        return None

    Ac = A - np.mean(A, axis=0)
    cov = (Ac.T @ Ac) / (N - 1)
    std_dev = np.std(A, axis=0, ddof=1, keepdims=True)
    deno = np.outer(std_dev, std_dev.T)
    return cov / deno