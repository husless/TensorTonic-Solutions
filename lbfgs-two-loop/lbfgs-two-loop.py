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
    q = list(grad)
    m = len(s_list)

    if m == 0:
        return [-gi for gi in g]

    alpha = [0 for _ in range(m)]
    rho = [0 for _ in range(m)]
    
    
    for i in range(m-1, -1, -1):
        si = s_list[i]
        yi = y_list[i]
        rho_i = 1.0 / _dot(yi, si)
        rho[i] = rho_i
        alpha_i = rho_i * _dot(si, q)
        alpha[i] = alpha_i
        
        q = [q_i - alpha_i*y_i for q_i, y_i in zip(q, yi)]

    ym_1 = y_list[m-1]
    gamma = _dot(s_list[m-1], ym_1) / _dot(ym_1, ym_1)
    r = [gamma*q_i for q_i in q]

    # 2-nd
    for i in range(m):
        yi = y_list[i]
        si = s_list[i]

        beta = rho[i] * _dot(yi, r)
        k1 = alpha[i] - beta
        r = [r_i + s_i*k1 for r_i, s_i in zip(r, si)]

    return [-r_i for r_i in r]