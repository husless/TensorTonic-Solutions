import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    AA = np.asarray(A)
    N, _ = AA.shape
    return sum(AA[i,i] for i in range(N))
