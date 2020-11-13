import pypermu
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
from datalog import DataLogger

DEVICE = 'cuda:0'


class Model(torch.nn.Module):

    def __init__(self, D_in, N, device=DEVICE):
        super(Model, self).__init__()
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


def marina2permu(marina):
    # TODO: Implement in rust for a faster func
    n = len(marina)
    e = list(range(n))
    permu = np.zeros(n, dtype=np.int64)
    for i, elem in enumerate(marina):
        permu[i] = e[elem]
        del e[elem]
    return permu


def sample(distribution, n_samples):
    samples = []
    for dist in distribution:
        samples.append(dist.sample([n_samples]))
    return torch.stack(samples).T


def log_probs(samples, distribution):
    log_probs = []
    for i, elems in enumerate(samples.T):
        log_probs.append(distribution[i].log_prob(elems))
    return torch.sum(torch.stack(log_probs), dim=0)


def entropy(distribution):
    return sum([d.entropy() for d in distribution])


def compute_loss(samples, distribution, fitness):
    logp = log_probs(samples, distribution)
    return (logp * fitness).mean()


@torch.no_grad()
def probability(marina, distribution):
    '''Computes the probability of a marina vector to be 
    sampled from a given marginal distribution.
    '''
    return np.prod([d.probs[marina[i]].item() for i, d in enumerate(distribution)])


if __name__ == '__main__':
    NOISE_LEN = 128
    N = 20
    N_SAMPLES = 64
    LR = .0003
    # ITERS = 2000
    ITERS = 4000
    C = 40

    problem = pypermu.problems.pfsp.Pfsp('../../instances/PFSP/tai20_5_8.fsp')

    dl = DataLogger()

    model = Model(NOISE_LEN, N, device=DEVICE)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)

    for it in range(ITERS):
        noise = torch.rand(NOISE_LEN).to(DEVICE)
        distribution = model.get_distribution(noise)
        samples = sample(distribution, 3)

        permus = [marina2permu(v) for v in samples.cpu()]
        fitness_list = torch.tensor(
            problem.evaluate(permus)).float().to(DEVICE)

        dl.push(fitness_list=fitness_list.cpu().numpy())

        fitness_list -= fitness_list.mean()

        h = entropy(distribution)
        dl.push(other={'entropy': h.item()})

        optimizer.zero_grad()  # clear gradient buffers
        loss = compute_loss(samples, distribution, fitness_list) - C*h
        loss.backward()  # update gradient buffers
        optimizer.step()  # update model's parameters

        dl.push(other={'loss': loss.item()})
        dl.push(other={'best sol. prob.': probability(
            samples[fitness_list.argmin()], distribution)})
        dl.push(other={'worst sol. prob.': probability(
            samples[fitness_list.argmax()], distribution)})

        print(it+1, '/', ITERS, end=' ')
        dl.print()

    dl.plot()
