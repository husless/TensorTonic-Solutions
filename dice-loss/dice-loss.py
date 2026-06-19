import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    P = np.asarray(p)
    Y = np.asarray(y)
    if P.ndim == 2:
        P = P.flatten()
        Y = Y.flatten()

    dice = (2*np.dot(P,Y) + eps) / (np.sum(P) + np.sum(Y) + eps)
    return 1 - dice