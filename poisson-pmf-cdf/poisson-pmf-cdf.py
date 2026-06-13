import numpy as np


def factorial(k: int) -> float:
    return np.exp(np.sum(np.log(np.arange(1, k+1))))


def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    k1 = np.exp(-lam)
    pmf = k1 * (lam ** k) / factorial(k)
    cdf = sum(
        k1 * (lam ** i) / factorial(i)
        for i in range(k+1)
    )
    return pmf, cdf