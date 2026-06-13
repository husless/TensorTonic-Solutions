import math


def cosine_annealing_schedule(
    base_lr: float,
    min_lr: float,
    total_steps: int,
    current_step: int
) -> float:
    """
    Compute the learning rate using cosine annealing.
    """
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos((current_step*math.pi)/total_steps))