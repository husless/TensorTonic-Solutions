import math


def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    n = len(predictions)
    pt = (
        p if t==1 else 1-p
        for p, t in zip(predictions, targets)
    )
    loss =[
        -alpha*((1-p)**gamma)*math.log(p)
        for p in pt 
    ]
    return sum(loss)/n