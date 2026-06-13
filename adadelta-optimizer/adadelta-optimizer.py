import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # squared gradient average
    E_g2_t = np.asarray(E_grad_sq)
    gt = np.asarray(grad)
    
    E_grad_sq_new = rho*E_g2_t + (1-rho)*(gt*gt)

    # weight update
    E_dw_sq_t = np.asarray(E_update_sq)
    delta_wt = - np.sqrt(E_dw_sq_t + eps)*gt / np.sqrt(E_grad_sq_new + eps)

    # squared upadte average
    E_dw_sq_t_new = rho*E_dw_sq_t + (1-rho)*(delta_wt**2)

    # weight
    wt = np.asarray(w) + delta_wt
    return wt, E_grad_sq_new, E_dw_sq_t_new