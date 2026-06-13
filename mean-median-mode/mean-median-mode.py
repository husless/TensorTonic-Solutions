import numpy as np
from collections import Counter


def _find_modes(x: list[float]) -> list[float]:
    if len(x) == 0:
        return []

    c = Counter(x)
    max_frequency = c.most_common(1)[0][1]
    return [item for item, count in c.items() if count == max_frequency]


def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    mean = np.mean(x)
    median = np.median(x)

    mode = min(_find_modes(x))
    return mean, median, mode
    