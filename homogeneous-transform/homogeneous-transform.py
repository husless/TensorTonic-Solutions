import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    p = np.asarray(points).reshape((-1,3))
    N, _ = p.shape
    p_h = np.hstack((p, np.ones((N,1))))
    t = (T @ p_h.T).T
    t1 = t[:,:-1]
    if N == 1:
        return t1.ravel()
    return t1
    