import torch
import numpy as np


def marina2permu(marina):
    # TODO: Implement in rust for a faster func
    n = len(marina)
    e = list(range(n))
    permu = np.zeros(n, dtype=np.int64)
    for i, elem in enumerate(marina):
        permu[i] = e[elem]
        del e[elem]
    return permu


def entropy(distribution, reduction='sum'):
    h = torch.stack([d.entropy() for d in distribution])
    if reduction == 'sum':
        return h.sum()
    elif reduction == 'none':
        return h


def probability(marina, distribution):
    '''Computes the probability of a marina vector to be
    sampled from a given marginal distribution.
    '''
    return np.prod([d.probs[marina[i]].item() for i, d in enumerate(distribution)])


def distr_to_square(distribution):
    matrix = []
    for i, d in enumerate(distribution):
        matrix.append(np.append(d.probs.cpu().detach().numpy(), [0]*i))

    return np.array(matrix)


def permu2inverse(permu):
    return np.array([np.where(permu == e)[0][0] for e in range(len(permu))])
