import math

def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    n = len(values)
    k = window_size
    result = [0 for _ in range(n-k+1)]

    for i in range(n-k+1):
        mu = sum(values[i+j] for j in range(k)) / k
        result[i] = math.sqrt(sum((values[i+j]-mu)**2 for j in range(k))/k)

    return result