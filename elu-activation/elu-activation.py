import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    return [
        alpha*(np.exp(z) - 1) if z < 0 else z
        for z in x
    ]