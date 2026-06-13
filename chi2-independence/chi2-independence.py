import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    row_total = np.sum(C, axis=1)
    col_total = np.sum(C, axis=0)
    total = np.sum(C)
    expected = np.outer(row_total, col_total) / total
    chi2 = np.sum((C - expected)**2 / expected)
    return chi2, expected