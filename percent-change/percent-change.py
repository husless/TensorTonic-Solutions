import itertools

def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    return [
        (s1-s0)/s0 if s0 != 0 else 0.0
        for s0, s1 in itertools.pairwise(series)
    ]