def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    x = list(series)
    n = len(series)
    for d in range(1, 1+order):
        for i in range(1, n-d+1):
            x[i-1] = x[i] - x[i-1]

    return x[:-d]