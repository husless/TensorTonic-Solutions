import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    yy = np.asarray(y, dtype=float).reshape((-1,1))
    XX = np.asarray(X, dtype=float)
    if XX.ndim == 1:
        XX = XX.reshape((-1,1))

    N,D = XX.shape
    scale = 1.0 / N
    
    w = np.zeros((D,1), dtype=float)
    b = 0

    for _ in range(steps):
        p = _sigmoid(XX @ w + b)
        diff = p - yy
        w = w - lr * scale * (XX.T @ diff)
        b = b - lr * scale * np.sum(diff)

    return w.ravel(), b