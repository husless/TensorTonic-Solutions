import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    gt = np.asarray(grad)
    
    # forst moment
    mt = beta1*np.asarray(m) + (1-beta1)*gt

    # second moment
    vt = beta2*np.asarray(v) + (1-beta2)*(gt**2)

    # AdamW
    w1 = np.asarray(w)
    wt = w1 - lr*(weight_decay*w1 + mt/(eps + np.sqrt(vt)))

    return wt, mt, vt