import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if not np.allclose(1.0, np.sum(p), 1.0E-6):
        raise ValueError("probabilities not sum to 1")
    X = np.asarray(x)
    P = np.asarray(p)
    if X.shape != P.shape:
        raise ValueError("shape of x and p mismatch")
    return np.dot(x, p)
