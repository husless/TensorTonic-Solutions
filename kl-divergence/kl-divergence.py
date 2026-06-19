import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    P = np.asarray(p)
    Q = np.asarray(q) + eps
    kl = np.where(P>0, P*np.log(P/Q), 0).sum()
    return kl
    