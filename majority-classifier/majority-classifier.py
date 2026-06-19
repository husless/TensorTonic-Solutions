import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    labels, counts = np.unique(y_train, return_counts=True)
    
    # index of most frequent class
    idx = np.argmax(counts)

    x_t = np.asarray(X_test)
    if x_t.size == 0:
        return []

    return np.full(x_t.shape, labels[idx], dtype=int)