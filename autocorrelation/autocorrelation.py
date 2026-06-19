def autocorrelation(series: list[float], max_lag: int):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    n = len(series)
    x_bar = sum(series) / n
    var = sum((x-x_bar)**2 for x in series)
    cor = [1.0] + [0.0]*max_lag
    if var < 1.0E-10:
        return cor
    for k in range(1, max_lag+1):
        cor[k] = sum((series[t]-x_bar)*(series[t+k]-x_bar) for t in range(n-k)) / var
    return cor