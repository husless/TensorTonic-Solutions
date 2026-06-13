import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    X = np.asarray(x)
    n = X.size
    mean = np.mean(X)
    s = np.std(X, ddof=1)
    t = (mean - mu0) * np.sqrt(n) / s
    return t