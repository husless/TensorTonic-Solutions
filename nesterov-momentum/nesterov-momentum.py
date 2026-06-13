import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    w1 = np.asarray(w)
    v1 = np.asarray(v)
    g = np.asarray(grad)
    
    # Look Ahead Position
    v_new = lr*g + momentum * v1
    w_new = w1 - v_new
    return w_new, v_new