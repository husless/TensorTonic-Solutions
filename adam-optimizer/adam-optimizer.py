import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    gt = np.asarray(grad)

    # first moment
    mt = beta1*np.asarray(m) + (1-beta1)*gt

    # second moment
    vt = beta2*np.asarray(v) + (1-beta2)*(gt**2)

    # bias correction
    mt_h = mt / (1-beta1**t)
    vt_h = vt / (1-beta2**t)

    # parameter update
    param_new = param - lr * mt_h / (eps + np.sqrt(vt_h))

    return param_new, mt, vt