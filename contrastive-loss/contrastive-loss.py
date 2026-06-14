import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1:
        D = a.shape[0]
        a = a.reshape((-1,D))
        b = b.reshape((-1,D))

    y = np.asarray(y)
    d = np.linalg.norm(a-b, axis=1)
    loss = y * (d**2) + (1-y)*np.maximum(0, margin-d)**2
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    else:
        raise ValueError("unkown reduction")
    