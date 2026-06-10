import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    AA = np.asarray(A)
    m, n = AA.shape
    t = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            t[j,i] = AA[i,j]
    return t
