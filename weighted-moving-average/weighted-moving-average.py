def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    n = len(values)
    k = len(weights)
    w = sum(weights)
    return [
        sum(wi*x for wi, x in zip(weights, values[i:i+k])) / w
        for i in range(0, n-k+1)
    ]