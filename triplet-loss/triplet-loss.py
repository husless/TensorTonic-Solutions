import numpy as np


def _dist(x, y):
    return np.sum((x - y)**2, axis=1)


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    a = np.asarray(anchor)
    p = np.asarray(positive)
    n = np.asarray(negative)
    if a.ndim == 1:
        D = a.shape[0]
        a = a.reshape((-1,D))
        p = p.reshape((-1,D))
        n = n.reshape((-1,D))

    loss = np.maximum(0, _dist(a,p) - _dist(a,n) + margin)
    return float(loss.mean())
