def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    n = len(returns)
    w = [0 for _ in range(n)]

    com = 1
    for i, ri in enumerate(returns):
        com *= 1 + ri
        w[i] = com - 1

    return w
