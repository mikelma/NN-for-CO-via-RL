import models
import torch
import utils
from torch.optim import Adam
from datalog import DataLogger
import pypermu

import matplotlib.pyplot as plt
from timeit import default_timer as timer


INSTANCE = '../../instances/PFSP/tai20_5_8.fsp'
N = 20
DEVICE = 'cpu'
NOISE_LEN = 128
LR = .0003
N_SAMPLES = 64
BATCH_SIZE = 32
ITERS = 1000
C = 40

problem = pypermu.problems.pfsp.Pfsp(INSTANCE)
model = models.SimpleModelBatched(NOISE_LEN, N, device=DEVICE)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

min_fitness_log = []
best_fitness_log = []
mean_fitness_log = []
for it in range(ITERS):
    noise = torch.rand((BATCH_SIZE, NOISE_LEN)).to(DEVICE)

    distribution, samples, logp = model.get_samples_and_logp(noise, N_SAMPLES)

    permus = [pypermu.utils.transformations.marina2permu_batched(
        b) for b in samples.cpu().numpy()]

    fitness_list = torch.tensor([problem.evaluate(batch)
                                 for batch in permus]).float().to(DEVICE)

    ############
    min_fitness_log.append(fitness_list.min().item())
    mean_fitness_log.append(fitness_list.mean().item())
    if len(best_fitness_log) == 0 or best_fitness_log[-1] > min_fitness_log[-1]:
        best_fitness_log.append(min_fitness_log[-1])
    else:
        best_fitness_log.append(best_fitness_log[-1])
    print('{}/{} min: {} best:{}'.format(it+1, ITERS,
                                         min_fitness_log[-1], best_fitness_log[-1]))
    plt.cla()
    plt.plot(range(it+1), mean_fitness_log, label='mean fitness')
    plt.plot(range(it+1), min_fitness_log, label='min fitness')
    plt.plot(range(it+1), best_fitness_log, label='best fitness')
    plt.legend()
    plt.pause(.001)
    ############

    fitness_list -= fitness_list.mean()

    # h = models.batched_entropy(distribution)

    optimizer.zero_grad()  # clear gradient buffers

    # loss = ((logp * fitness_list).mean(1) - C * h).mean()
    loss = (logp * fitness_list).mean()

    loss.backward()  # update gradient buffers
    optimizer.step()  # update model's parameters

plt.show()
