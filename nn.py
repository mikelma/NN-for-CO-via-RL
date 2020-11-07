import pypermu
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

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


def compute_loss(samples, distribution, fitness):
    logp = log_probs(samples, distribution)
    return (logp * fitness).mean()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    NOISE_LEN = 128
    N = 20
    N_SAMPLES = 64
    LR = .0003
    ITERS = 2000

    problem = pypermu.problems.pfsp.Pfsp('../../instances/PFSP/tai20_5_8.fsp')

    model = Model(NOISE_LEN, N, device=DEVICE)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)

    log_min = []
    log_best = []
    log_loss = []
    for it in range(ITERS):
        noise = torch.rand(NOISE_LEN).to(DEVICE)
        distribution = model.get_distribution(noise)
        samples = sample(distribution, 3)

        permus = [marina2permu(v) for v in samples.cpu()]
        fitness_list = torch.tensor(
            problem.evaluate(permus)).float().to(DEVICE)

        log_min.append(torch.min(fitness_list).item())
        log_best.append(np.min(log_min))

        fitness_list -= fitness_list.mean()

        optimizer.zero_grad()  # clear gradient buffers
        loss = compute_loss(samples, distribution, fitness_list)
        loss.backward()  # update gradient buffers
        optimizer.step()  # update model's parameters

        print(it, ', loss; {:10.3f}'.format(loss.item()),
              ', min fitness: {:10.3f}'.format(log_min[-1]))
        log_loss.append(loss.item())

        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(log_min)), log_min, label='min fitness')
        plt.plot(range(len(log_best)), log_best, label='best fitness')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(range(len(log_loss)), log_loss, label='loss')
        plt.legend()
        plt.pause(.001)
