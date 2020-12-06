import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch


class SimpleModel(torch.nn.Module):

    def __init__(self, D_in, N, device='cuda:0'):
        super(SimpleModel, self).__init__()
        self.n = N  # length of the inversion vector to sample
        self.dev = device

        # create shared layer
        self.l1 = torch.nn.Linear(D_in, 512)
        # an output layer for each position in the solutions
        self.out_layers = torch.nn.ModuleList(
            [torch.nn.Linear(512, self.n-i) for i in range(N)])

    def forward(self, x):
        x = F.relu(self.l1(x))
        out_list = [layer(x) for layer in self.out_layers]
        return out_list

    def get_distribution(self, x):
        weights = self.forward(x)
        distribs = [Categorical(logits=w) for w in weights]
        return distribs


class MultiHeadModel(torch.nn.Module):

    def __init__(self, D_in, N, device='cuda:0'):
        super(MultiHeadModel, self).__init__()
        self.n = N  # length of the inversion vector to sample
        self.dev = device

        # create shared layer
        self.l1 = torch.nn.Linear(D_in, 512)
        # an output layer for each position in the solutions
        self.policy_head = torch.nn.ModuleList(
            [torch.nn.Linear(512, self.n-i) for i in range(N)])
        # the state-value prediction head
        self.value_head = torch.nn.Linear(512, 1)

    def forward(self, x):
        # shared layer
        shared_out = F.relu(self.l1(x))

        # policy head output
        policy = [layer(shared_out) for layer in self.policy_head]

        # state-value prediction
        value = self.value_head(shared_out)

        return policy, value

    def get_distribution_and_value(self, x):
        policy_weights, value = self.forward(x)
        policy = [Categorical(logits=w) for w in policy_weights]
        return policy, value


def sample_batched(distribution, n_samples):
    print('num samples: 5')
    samples = []
    for dist in distribution:
        samples.append(dist.sample([5]))
    return torch.stack(samples, dim=1).permute(2, 0, 1)


def sample(distribution, n_samples):
    samples = []
    for dist in distribution:
        samples.append(dist.sample([n_samples]))
    return torch.stack(samples).T


def log_probs(samples, distribution):
    '''Returs the log probabilities of the samples given a probability distribution.
    '''
    log_probs = []  # logprob of the element in the i-th position of each sample
    for i, elems in enumerate(samples.T):
        log_probs.append(distribution[i].log_prob(elems))
    return torch.sum(torch.stack(log_probs), dim=0).T


def log_probs_unprocessed(samples, distribution):
    log_probs = []  # logprob of the element in the i-th position of each sample
    for i, elems in enumerate(samples.T):
        log_probs.append(distribution[i].log_prob(elems))
    return torch.stack(log_probs).mean(1)
