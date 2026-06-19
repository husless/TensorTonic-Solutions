import math


def label_smoothing_loss(
    predictions: list[float],
    target: int,
    epsilon: float
) -> float:
    """
    Compute cross-entropy loss with label smoothing.
    """
    K = len(predictions)
    q = [
        1 - epsilon + epsilon / K if i == target else epsilon / K
        for i in range(K)
    ]
    loss = - sum(q_i*math.log(p_i) for q_i, p_i in zip(q, predictions) )
    return loss
    