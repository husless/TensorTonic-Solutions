import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    A = np.asarray(matrix)
    if len(A) == 0:
        return A

    if axis is not None and A.ndim <= axis:
        return None

    if A.ndim > 2:
        return None

    if norm_type == 'l1':
        norm_ord = 1
    elif norm_type == 'l2':
        norm_ord = None
    elif norm_type == 'max':
        norm_ord = np.inf
    else:
        return None

    norm = np.linalg.norm(A, ord=norm_ord, axis=axis, keepdims=True)
    norm[norm == 0] = np.inf
    return A / norm