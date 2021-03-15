import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch


class SimpleModel(torch.nn.Module):
    name = 'SimpleModel 1'

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


class SimpleModelBatched(torch.nn.Module):
    name = 'SimpleModelBatched 1'

    def __init__(self, D_in, N, device='cuda:0'):
        super(SimpleModelBatched, self).__init__()
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

    def get_samples_and_logp(self, x, n_samples):
        logits = self.forward(x)
        distribs = []
        samples = []
        logps = []
        for l in logits:
            d = Categorical(logits=l)
            sample = d.sample((n_samples,))
            samples.append(sample)
            distribs.append(d)
            logps.append(d.log_prob(sample))

        samples = torch.stack(samples, dim=0).T
        # logps = torch.stack(logps, dim=0).T.sum(2)
        logps = torch.stack(logps, dim=0).T
        return distribs, samples, logps

    # def get_samples_and_logp_and_intermediate(self, x, n_samples):
    #     logits, shared_out = self.forward(x)
    #     distribs = []
    #     samples = []
    #     logps = []
    #     for l in logits:
    #         d = Categorical(logits=l)
    #         sample = d.sample((n_samples,))
    #         samples.append(sample)
    #         distribs.append(d)
    #         logps.append(d.log_prob(sample))

    #     samples = torch.stack(samples, dim=0).T
    #     logps = torch.stack(logps, dim=0).T
    #     return distribs, samples, logps, shared_out.mean(0)


class TinyBatched(torch.nn.Module):
    name = 'TinyBatched'

    def __init__(self, D_in, N, device='cuda:0'):
        super(TinyBatched, self).__init__()
        self.n = N  # length of the inversion vector to sample
        self.dev = device

        # an output layer for each position in the solutions
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(D_in, N-i) for i in range(N)])

    def forward(self, x):
        return [layer(x) for layer in self.layers]

    def get_samples_and_logp(self, x, n_samples):
        logits = self.forward(x)
        distribs = []
        samples = []
        logps = []
        for l in logits:
            d = Categorical(logits=l)
            sample = d.sample((n_samples,))
            samples.append(sample)
            distribs.append(d)
            logps.append(d.log_prob(sample))

        samples = torch.stack(samples, dim=0).T
        logps = torch.stack(logps, dim=0).T
        return distribs, samples, logps


class DeepBatched(SimpleModelBatched):
    name = 'DeepBatched'

    def __init__(self, D_in, N, device='cuda:0'):
        super(DeepBatched, self).__init__(D_in, N, device)

        # create additional layers
        self.l2 = torch.nn.Linear(512, 512)
        self.l3 = torch.nn.ModuleList(
            [torch.nn.Linear(512, 512) for i in range(N)])

        def forward(self, x):
            # shared layers
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))

            out_list = [layer(x) for layer in self.out_layers]
            out_list = []
            for i in range(self.n):
                interm = F.relu(self.l3[i](x))
                out_list.append(self.out_layers[i](interm))
            return out_list


class BatchedResidual(SimpleModelBatched):
    name = 'BatchedResidual'

    def __init__(self, D_in, N, device='cuda:0'):
        super(BatchedResidual, self).__init__(D_in, N, device)
        # identity tensors do not require gradient
        self.identity = [torch.as_tensor(
            [1.] + [0.]*(N-i-1), device=device) for i in range(N)]

    def forward(self, x):
        logits = super().forward(x)
        # relu activation is needed to avoid negative values
        return [F.relu(self.identity[i] + logits[i]) for i in range(self.n)]

    def get_samples_and_logp(self, x, n_samples):
        out = self.forward(x)
        distribs = []
        samples = []
        logps = []
        for probs in out:
            # probabilities will be normalized to sum 1
            d = Categorical(probs=probs)
            # collect new samples
            sample = d.sample((n_samples,))
            samples.append(sample)
            distribs.append(d)
            # get log probabilities of samples
            logps.append(d.log_prob(sample))

        samples = torch.stack(samples, dim=0).T
        logps = torch.stack(logps, dim=0).T
        return distribs, samples, logps


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
    return torch.stack(log_probs).T


def batched_entropies(distribs):
    '''Given a batch of distributions, returns a batched entropies of all N distributions.
    In other words, returns entriopy tendsor of shape: (batch num., num. distribs.).
    '''
    # h = (batch, entropy of each distribution of the batch)
    h = torch.stack([d.entropy() for d in distribs]).T
    return h
