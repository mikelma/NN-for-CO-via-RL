import utils
import models
import numpy as np
import torch


def compute_l2(samples, distribution, fitness, entropy, C):
    logp = models.log_probs(samples, distribution)
    return (logp * fitness).mean() - C*entropy


def compute_l3(samples, distribution, fitness, entropy, N):
    max_entropy = torch.tensor(
        sum([i*(1/i)*np.log(1/i) for i in range(1, N+1)]))
    logp = models.log_probs(samples, distribution)
    return -(logp * fitness).mean()*(entropy/max_entropy)
