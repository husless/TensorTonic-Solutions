import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    z = np.asarray(x)
    return 0.5 * z * (1 + np.vectorize(math.erf)(0.5*math.sqrt(2)*z))
    
