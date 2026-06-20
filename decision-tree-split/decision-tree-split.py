import numpy as np


def _gini(y):
    n = len(y)
    if n == 0:
        return 0
        
    labels, counts = np.unique(y, return_counts=True)
    p = counts / n
    impurity = 1 - np.sum(p**2)
    return impurity


def _weighted_gini(left_y, right_y):
    """Calculate the impurity from a split."""
    
    # Weight the impurity of children by their sample size
    n_l, n_r = len(left_y), len(right_y)
    
    if n_l == 0 or n_r == 0:
        return 0
        
    return (n_l * _gini(left_y) + n_r * _gini(right_y)) / (n_l + n_r)



def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.

    X: (n_samples, n_features), feature matrix
    y: (n_samples,) labels
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n, d = X.shape
    
    best_gain = -float('inf')
    best_idx, best_thr = 0, float('inf')

    parent_gini = _gini(y)

    for feat_idx in range(d):
        X_column = X[:, feat_idx]
        # Test every unique value in the column as a threshold
        vals = np.unique(X_column) # sorted by default

        for thr1, thr2 in zip(vals, vals[1:]):
            threshold = (thr1 + thr2) / 2
            # Generate mask arrays for the split
            left_mask = X_column <= threshold
            right_mask = ~left_mask
            
            left_y, right_y = y[left_mask], y[right_mask]
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            
            # Calculate Information Gain
            gain = parent_gini - _weighted_gini(left_y, right_y)
            
            # Keep track of the best split seen so far
            # the feature and threshold with the highest gain.
            # Break ties by smallest feature index, then smallest threshold.
            if gain > best_gain:
                best_gain = gain
                best_idx = feat_idx
                best_thr = threshold
            elif gain == best_gain and threshold < best_thr:
                best_idx = feat_idx
                best_thr = threshold

    return best_idx, best_thr