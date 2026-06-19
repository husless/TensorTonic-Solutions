def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    v0 = values[0]
    lo = 0
    up = 0
    result = list(values)
    for i, xi in enumerate(values):
        if xi is None:
            if not (lo < up):
                j = i + 1
                while values[j] is None:
                    j += 1

                up = j
            result[i] = v0 + (values[up]-v0)*(i-lo)/(up-lo)
        else:
            v0, lo = xi, i
    return result
    
            