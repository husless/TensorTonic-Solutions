def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    k = window_size
    n = len(values)
    s = sum(values[:k])
    res = [0 for _ in range(n-k+1)]
    res[0] = s/k
    for i in range(1, n-k+1):
        s += values[i+k-1] - values[i-1]
        res[i] = s / k

    return res