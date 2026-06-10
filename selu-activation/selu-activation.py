import numpy as np

def selu(
    x: list[float],
    lam: float = 1.0507009873554804934193349852946,
    alpha: float = 1.6732632423543772848170429916717
):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    g = (
        alpha * (np.exp(z) - 1) if z < 0 else z
        for z in x
    )
    return [round(lam*z, 4) for z in g]
    