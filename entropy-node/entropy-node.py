import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    n = len(y)
    if n == 0:
        return 0.0

    labels, counts = np.unique(y, return_counts=True)
    p = counts / n
    entropy = - np.sum(p*np.log2(p))
    return entropy