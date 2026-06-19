import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    z1 = np.asarray(Z1)
    z2 = np.asarray(Z2)
    s0 = (z1 @ z2.T) / temperature
    s = s0 - np.max(s0)
    loss = - np.mean(np.log(np.exp(np.diag(s))/np.sum(np.exp(s), axis=1)))
    return loss
    