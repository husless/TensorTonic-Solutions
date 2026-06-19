import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    P = np.asarray(p)
    Y = np.asarray(y)
    loss = ((1-P)**gamma)*Y*np.log(P) + (P**gamma)*(1-Y)*np.log(1-P)
    return -loss.mean()