
def _grad(a,b,x):
    return 2*a*x + b


def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    for _ in range(steps):
        x = x - lr * _grad(a,b,x)

    return x