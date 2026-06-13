import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    x_train = np.asarray(X_train)
    x_test = np.asarray(X_test)
    d = x_test.ndim
    if d == 1:
        x_train = x_train.reshape(-1,1)
        x_test = x_test.reshape(-1,1)

    diff = x_test[:, np.newaxis, :] - x_train[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    ind = np.argsort(dist_sq, axis=-1)[:, :k]
    r, n = ind.shape

    if n < k:
        return np.hstack((ind, -np.ones((r, k-n), dtype=int)))
    
    return ind
    