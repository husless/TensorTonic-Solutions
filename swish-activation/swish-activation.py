import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    z = np.asarray(x)
    return z / (1 + np.exp(-z))