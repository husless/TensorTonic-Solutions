import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    q = 1 - p
    pmf = comb(n, k)* (p ** k) * (q**(n-k))
    cdf = sum(
        comb(n, i) * (p ** i) * (q ** (n-i))
        for i in range(k+1)
    )
    return pmf, cdf