import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y)
    mask = np.asarray(split_mask, dtype=bool)
    yl = y[mask]
    yr = y[~mask]
    n1 = len(yl)
    n2 = len(yr)
    if n1 >0 and n2 > 0:
        n = n1 + n2
        return _entropy(y) - (n1*_entropy(yl) + n2*_entropy(yr)) / n
    else:
        return 0.0
