import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.

    y_true: (n_samples,), index of valid class in y_pred
    y_pred: (n_samples, n_labels), probabilities, each row sums to 1
    """
    yt = np.asarray(y_true)
    log_yp = np.log(np.asarray(y_pred))
    n_samples, _ = log_yp.shape
    rows = np.arange(n_samples)
    ce = log_yp[rows, yt]
    return -np.mean(ce)
    