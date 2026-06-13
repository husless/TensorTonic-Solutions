import numpy as np

def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def lbfgs_direction(
    grad: list[float],
    s_list: list[list[float]],
    y_list: list[list[float]]
) -> list[float]:
    """
    grid: (n, )
    s_list: (m, n)
    y_list: (m, n)
    
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    # 1-st
    s_mat = np.asarray(s_list, dtype=float)
    y_mat = np.asarray(y_list, dtype=float)

    rho = 1.0 / np.einsum('ij,ij->i', y_mat, s_mat)

    m, _ = s_mat.shape
    q = np.array(grad, dtype=float)
    alpha = np.zeros(m, dtype=float)

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * np.dot(s_mat[i], q)
        q -= alpha[i] * y_mat[i]

    gamma = np.dot(s_mat[-1], y_mat[-1]) / np.dot(y_mat[-1], y_mat[-1])
    r = gamma * q
    
    # 2-nd
    for i in range(m):
        beta = rho[i] * np.dot(y_mat[i], r)
        r += (alpha[i] - beta) * s_mat[i]

    return -r