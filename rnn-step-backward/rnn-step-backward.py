import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    x_t = np.asarray(cache[0])
    h_prev = np.asarray(cache[1])
    h_t = np.asarray(cache[2])
    W = np.asarray(cache[3])
    U = np.asarray(cache[4])
    b = np.asarray(cache[5])

    # 1. Apply chain rule with upstream gradient dh
    dtanh = dh * (1 - h_t**2)
    
    # 2. Transpose weights to project gradients back to inputs
    dx_t = W.T @ dtanh
    dh_prev = U.T @ dtanh
    
    # 3. Align outer product order to match weight dimensions
    dW = np.outer(dtanh, x_t)
    dU = np.outer(dtanh, h_prev)
    db = dtanh

    return dx_t, dh_prev, dW, dU, db
    
    
