import pypermu
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np


class Model(torch.nn.Module):

    def __init__(self, D_in, N, device='cuda:0'):
        super(Model, self).__init__()
        self.n = N  # length of the inversion vector to sample
        self.dev = device
        self.l1 = torch.nn.Linear(D_in, 512)
        self.l2 = torch.nn.Linear(512, N*N)

        self.mask = torch.tensor(
            [[1 if N-j > i else 0 for j in range(N)] for i in range(N)],
            requires_grad=False
        ).to(self.dev)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x

    def get_distribution(self, x):
        logits = self.forward(x).view((self.n, self.n))
        probs = torch.softmax(logits, dim=1)
        # mask the logits in order to give 0 prob. to the values that cannot be sampled
        # for each position
        probs_masked = probs * self.mask
        # the probabilities will be normalized by Categorical automatically
        return Categorical(probs=probs_masked)

    def sample(self, x, n_samples, return_entopy=False):
        d = self.get_distribution(x)
        if return_entopy:
            print(d.entropy())
            quit()
        else:
            return d.sample([n_samples])


def marina2permu(marina):
    # TODO: Implement in rust for a faster func
    n = len(marina)
    e = list(range(n))
    permu = np.zeros(n, dtype=np.int64)
    for i, elem in enumerate(marina):
        permu[i] = e[elem]
        del e[elem]
    return permu


def compute_loss(model, x, v, fitness):
    logp = model.get_distribution(x).log_prob(v).sum(1)
    # print('sum testing instead of mean')
    # return (logp * fitness).sum()
    return (logp * fitness).mean()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    NOISE_LEN = 128
    N = 20
    DEVICE = 'cuda:0'
    N_SAMPLES = 64
    LR = .001
    # LR = .00005
    ITERS = 2000
    # BATCH_SIZE = 1024  # = num. of samples
    # LR = .001

    # qap = pypermu.problems.qap.Qap('instances/tai20b.dat')
    problem = pypermu.problems.pfsp.Pfsp('../instances/PFSP/tai20_5_8.fsp')

    model = Model(NOISE_LEN, N, device=DEVICE)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)

    min_fitness = []
    mean_fitness = []
    entropy = []
    loss_log = []
    probs = None
    best_sol_fitness = None
    for it in range(ITERS):
        # generate random noise as input for the model
        noise = torch.rand(NOISE_LEN).to(DEVICE)
        # noise = torch.rand((BATCH_SIZE, NOISE_LEN)).to(DEVICE)
        # sample inversion vectors from the model
        v = model.sample(noise, N_SAMPLES).cpu().numpy()
        # codify inversion vectors as permutations
        permus = np.apply_along_axis(marina2permu, 1, v)

        # evaluate permutations
        fitness = problem.evaluate(permus)

        # ---- stats ---- #
        min_fitness.append(np.min(fitness))
        if best_sol_fitness is None:
            best_sol_fitness = min_fitness[-1]
        elif best_sol_fitness > min_fitness[-1]:
            best_sol_fitness = min_fitness[-1]

        mean_fitness.append(np.mean(fitness))
        print(it+1, '/', ITERS, 'mean: ', mean_fitness[-1],
              ', min: {:.2E}'.format(min_fitness[-1]),
              ', best: ', best_sol_fitness)

        plt.clf()
        # plt.subplot(2, 2, 1)
        plt.plot(range(len(min_fitness)), min_fitness, label='min fitness')
        plt.plot(range(len(mean_fitness)), mean_fitness, label='mean fitness')
        plt.legend()

        # plt.subplot(2, 2, 2)
        # plt.plot(range(len(loss_log)), loss_log, color='r')
        # plt.title('Loss value')

        # plt.subplot(2, 2, 3)
        # if it % 50 == 0:
        #     with torch.no_grad():
        #         distr = model.get_distribution(noise)
        #         entropy.append(distr.entropy().sum().item())
        #         probs = distr.probs.cpu().numpy()
        # plt.imshow(probs)
        # plt.title('Prob. distribution defined over the inversion vector space')

        # plt.subplot(2, 2, 4)
        # plt.plot(range(len(entropy)), entropy)
        # plt.title('Entropy')
        plt.pause(.001)
        # -------------- #

        fitness = torch.tensor(fitness, dtype=torch.float,
                               requires_grad=False).to(DEVICE)
        # fitness = (fitness - fitness.mean()) / fitness.std() # normalize
        fitness -= fitness.mean()

        sampled = torch.from_numpy(v).to(DEVICE)

        optimizer.zero_grad()  # clear gradient buffers
        loss = compute_loss(model, noise, sampled, fitness)
        loss_log.append(loss.item())
        loss.backward()  # update gradient buffers
        optimizer.step()  # update model's parameters

    plt.show()
