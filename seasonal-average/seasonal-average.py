def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    n = len(series)

    seasonal = ([series[i] for i in range(p, n, period)] for p in range(period))
    return [
        sum(s) / len(s)
        for s in seasonal
    ]