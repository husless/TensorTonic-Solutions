import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    X, is_1d = _as2d(x, -1)
    H_prev, _ = _as2d(h_prev, -1)

    zt = _sigmoid(X @ params["Wz"] + H_prev @ params['Uz'] + params['bz'])
    rt = _sigmoid(X @ params['Wr'] + H_prev @ params['Ur'] + params['br'])
    ht_bar = np.tanh(X @ params['Wh'] + (rt*H_prev) @ params['Uh'] + params['bh'])
    ht = (1-zt) * H_prev + zt * ht_bar
    if is_1d:
        return ht.squeeze()
    else:
        return ht
    