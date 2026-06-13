import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    z = np.asarray(x)
    pmf = np.where(z == 0, 1-p, p)
    mu = p
    var = p * (1-p)
    return (pmf, mu, var)