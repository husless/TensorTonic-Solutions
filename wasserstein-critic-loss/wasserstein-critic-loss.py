import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    mu_r = np.mean(real_scores)
    mu_f = np.mean(fake_scores)
    return mu_f - mu_r