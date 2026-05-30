import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    ytrue = np.asarray(y_true)
    ypred = np.asarray(y_pred)
    if np.all(ytrue==ytrue[0]):
        if np.all(ytrue==ypred):
            return 1.0
        else:
            return 0.0
    res = ytrue - ypred
    ss_res = np.dot(res, res)
    tot = ytrue - np.mean(ytrue)
    ss_tot = np.dot(tot,tot)
    return 1.0 - ss_res / ss_tot