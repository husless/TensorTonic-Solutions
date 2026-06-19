import numpy as np


def _gini(y):
    n = len(y)
    if n == 0:
        return 0

    label, count = np.unique(y, return_counts=True)
    p = count / len(y)
    return 1 - np.sum(p**2)


def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    n1 = len(y_left)
    n2 = len(y_right)
    n = n1 + n2
    if n == 0:
        return 0.0

    return (n1*_gini(y_left) + n2*_gini(y_right)) / n