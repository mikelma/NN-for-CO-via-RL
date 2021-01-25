import argparse
import torch
import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-size', metavar='N', type=int, nargs=1,
                        required=True, help='Size of the instance', choices=[20, 50, 100])
    parser.add_argument('--instance', metavar='FILE', type=str, nargs=1,
                        required=True, help='Path to the instance file')
    parser.add_argument('--log', type=str, nargs='?', default=False, const=True,
                        required=False, help='If this falg is set, the logger will be stored as a CSV')
    parser.add_argument('--wandb', type=str, nargs='?', default=False, const=True,
                        required=False, help='If this falg is provided, weight and biases will be used to track the experiment')
    args = parser.parse_args()

    write_log = False if args.log == False else True
    wandb_enable = False if args.wandb == False else True
    return args.instance[0], args.instance_size[0], write_log, wandb_enable


def marina2permu(marina):
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
