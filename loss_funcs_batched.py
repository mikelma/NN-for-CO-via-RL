import torch
import numpy as np


def mean_utility(fitness_list):
    return fitness_list - fitness_list.mean(1).view(-1, 1)


def loss_l1(fitness_list, logps, utility=mean_utility):
    u = utility(fitness_list)
    # log P(V) = sum(log P(V(i))), where i=0..(n-1)
    sample_logp = logps.sum(-1)
    return (sample_logp * u).mean()


def loss_l5(fitness_list, logps, distribs, gamma=1., utility=mean_utility, device='cpu'):
    gamma = torch.as_tensor(gamma, dtype=torch.float32, device=device)
    n = len(distribs)

    # compute Z vector (max entropy)
    Z = torch.as_tensor([-np.log(1/(n-i)) for i in range(n-1)], device=device)
    # entropy for each distribution of each batch. shape: (batch_size, N)
    H = torch.stack([d.entropy() for d in distribs]).T
    # rho: convergency vector, averaged across distribution dimension. shape: (batch_size)
    rho = (H[:, :-1] / Z).mean(1)

    # add a new dimension and ignore H(V_n), as it is always 0
    H = (H+gamma).view(H.size(0), H.size(1), 1)[:, :-1, :]
    # transpose is needed to match dimensions of H. Also, ignore H(V_n)
    logps = logps[:, :, :-1].transpose(1, 2)
    scaled_logps = (logps / H).sum(1)  # scaled logP of each sample

    u = utility(fitness_list)

    return ((u * scaled_logps).mean(1) * rho).mean()
