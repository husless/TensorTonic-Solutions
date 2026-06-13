import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # accumulate squared gradiennts
    gt = np.asarray(g)
    Gt = np.asarray(G) + gt*gt
    wt = np.asarray(w) - lr * gt / np.sqrt(Gt + eps)
    return wt, Gt