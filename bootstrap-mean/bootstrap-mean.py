import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    r = rng if rng is not None else np.randoom.default_rng()
    X = np.asarray(x)
    N = X.size
    b_samples = np.array([np.mean(X[r.integers(N, size=N)]) for _ in range(n_bootstrap)])
    alpha = 100 * 0.5 * (1 - ci)
    lower, upper = np.percentile(b_samples, [alpha, 100 - alpha])
    return b_samples, lower, upper
