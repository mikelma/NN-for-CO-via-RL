import utils
import models
import numpy as np
import torch


def compute_l2(samples, distribution, fitness, C):
    '''L2 loss function. Note: when the C parameter is set to 0,
    this loss is equal to the L1 loss function.
    '''
    fitness -= fitness.mean()  # mean based utility function

    h = utils.entropy(distribution)
    logp = models.log_probs(samples, distribution)
    return (logp * fitness).mean() - C*h


def compute_l3(samples, distribution, fitness, N):
    fitness -= fitness.mean()  # mean based utility function

    # calculate the maximum possible entropy of the distribution
    max_entropy = torch.tensor(
        sum([i*(1/i)*np.log(1/i) for i in range(1, N+1)]))
    # get logps and entropy
    logp = models.log_probs(samples, distribution)
    h = utils.entropy(distribution)
    return -(logp * fitness).mean()*(h/max_entropy)


def compute_l4(samples, distribution, fitness):
    fitness -= fitness.mean()  # mean based utility function

    h = utils.entropy(distribution, reduction='none')
    logp = models.log_probs_unprocessed(samples, distribution)

    # log and entropy of the las distribution P(v_n) is ignored,
    # as this values are always 0 and dividing 0 by itself
    # causes bad things to happen
    scaled_logps = (logp[:, :-1]/h[:-1]).sum(1)
    return (scaled_logps * fitness).mean()


def compute_l5(samples, distribution, fitness, gamma):
    fitness -= fitness.mean()  # mean based utility function

    gamma = torch.tensor(gamma)
    h = utils.entropy(distribution, reduction='none')
    n = h.size(0)

    max_h = torch.tensor([-np.log(1/i) for i in reversed(range(1, n+1))])
    convergency = h[:-1] / max_h[:-1]
    convergency = convergency[:-1].sum()

    logp = models.log_probs_unprocessed(samples, distribution)
    # log and entropy of the las distribution P(v_n) is ignored,
    # as this values are always 0 and dividing 0 by itself
    # causes bad things to happen
    scaled_logps = (logp[:, :-1]/(h[:-1]+gamma)).sum(1)

    return (scaled_logps * fitness * convergency).mean(), convergency, scaled_logps.mean()
