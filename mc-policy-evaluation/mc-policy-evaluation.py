import numpy as np

from numpy.typing import NDArray


def mc_policy_evaluation(
    episodes: list[list[tuple[int,float]]],
    gamma: float,
    n_states: int
) -> NDArray[float]:
    """
    episodes: list of episodes - each episode is list of (state, reward) tuples
    gamma: float - discount factor
    n_states: int - number of states (labeled 0..n_states-1)

    Returns: V (NumPy array of shape (n_states,))
    """
    # Initialize arrays to track total returns and visit counts
    total_returns = np.zeros(n_states)
    visit_counts = np.zeros(n_states)

    for episode in episodes:
        g = 0
        visited_states = set()

        # Iterate backward through the episode to calculate returns efficiently
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            g = reward + gamma * g

            # Check if this is the first visit to the state in this episode
            # We look ahead to see if the state appeared earlier in the episode
            earlier_states = [step[0] for step in episode[:t]]

            if state not in earlier_states:
                total_returns[state] += g
                visit_counts[state] += 1

    # Avoid division by zero for unvisited states
    v = np.zeros(n_states)
    visited = visit_counts > 0
    v[visited] = total_returns[visited] / visit_counts[visited]

    return v
