def _median(x: list[float], n: int) -> float:
    x1 = sorted(x)
    if n%2==1:
        return x1[n//2]
    return 0.5 * (x1[n//2-1] + x1[n//2])


def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    n = len(values)
    k = window_size
    return [
        _median(values[i:i+k], k)
        for i in range(n-k+1)
    ]
    