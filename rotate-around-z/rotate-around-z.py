import numpy as np


def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    a = np.cos(theta)
    b = np.sin(theta)
    R = np.array([[a,-b, 0],
                  [b, a, 0],
                  [0, 0, 1]])
    p = np.asarray(points).reshape((-1,3))
    N, _ = p.shape
    p1 = R @ p.T
    if N == 1:
        return p1.ravel()
    return p1.T