import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    gt = np.asarray(g)

    # running average
    st = beta*np.asarray(s) + (1-beta)*(gt**2)

    # parameters
    wt = w - lr * gt / np.sqrt(st+eps)
    return wt, st