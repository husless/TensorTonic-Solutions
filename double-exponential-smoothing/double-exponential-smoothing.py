def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    level_0 = series[0]
    trend_0 = series[1] - series[0]
    level = [level_0 for _ in series]
    for t, yt in enumerate(series[1:], start=1):
        lt = alpha*yt + (1-alpha)*(level[t-1]+trend_0)
        trend_0 = beta*(lt-level[t-1]) + (1-beta)*trend_0
        level[t] = lt

    return level