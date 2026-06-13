def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    n = len(values)
    ema = [0 for _ in range(n)]
    ema[0] = values[0]
    for i in range(1,n):
        ema[i] = alpha*values[i] + (1-alpha) * ema[i-1]

    return ema
