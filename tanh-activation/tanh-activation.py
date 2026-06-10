import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    a = np.exp(x)
    b = 1.0 / a
    return (a - b) / (a + b)