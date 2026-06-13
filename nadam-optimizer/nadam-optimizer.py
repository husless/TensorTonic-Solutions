import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    gt = np.asarray(grad)
    
    # 1-st moment
    mt = beta1*np.asarray(m) + (1-beta1)*gt

    # 2-nd moment
    vt = beta2*np.asarray(v) + (1-beta2)*(gt*gt)

    # Nesterov-Adjusted update
    wt = np.asarray(w) - lr*(beta1*mt + (1-beta1)*gt) / (np.sqrt(vt) + eps)

    return wt, mt, vt
