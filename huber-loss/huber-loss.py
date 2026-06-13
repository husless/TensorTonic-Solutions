import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    e_abs = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    l_delta = np.where(e_abs > delta, delta*(e_abs - 0.5*delta), 0.5*(e_abs**2))
    return l_delta.mean()