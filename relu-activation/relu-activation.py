import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    return np.maximum(0, x)