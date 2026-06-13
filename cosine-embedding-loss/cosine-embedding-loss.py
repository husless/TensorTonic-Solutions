import math


def _dot(x: list[float], y: list[float]) -> float:
    return sum(xi*yi for xi, yi in zip(x,y))


def cosine_embedding_loss(
    x1: list[float],
    x2: list[float],
    label: int,
    margin: float
) -> float:
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1_norm = math.sqrt(_dot(x1,x1))
    x2_norm = math.sqrt(_dot(x2,x2))
    cosine = _dot(x1,x2) / (x1_norm*x2_norm)
    if label == 1:
        return 1 - cosine
    elif label == -1:
        return max(0, cosine - margin)
    else:
        raise ValueError("unkown label")
    