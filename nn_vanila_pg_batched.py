import models
import torch
import utils
from torch.optim import Adam
from datalog import DataLogger
import pypermu
import matplotlib.pyplot as plt


INSTANCE = '../../instances/PFSP/tai20_5_8.fsp'
N = 20
DEVICE = 'cpu'
NOISE_LEN = 128
LR = .0003
N_SAMPLES = 64
BATCH_SIZE = 16
ITERS = 500
C = 40

problem = pypermu.problems.pfsp.Pfsp(INSTANCE)
model = models.SimpleModelBatched(NOISE_LEN, N, device=DEVICE)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

min_fitness_log = []
for it in range(ITERS):
    noise = torch.rand((BATCH_SIZE, NOISE_LEN)).to(DEVICE)
    distribution, samples, logp = model.get_samples_and_logp(noise, N_SAMPLES)

    permus = [[utils.marina2permu(v) for v in batch]
              for batch in samples.cpu()]
    fitness_list = torch.tensor([problem.evaluate(batch)
                                 for batch in permus]).float().to(DEVICE)

    print('{}/{} min fitness: {}'.format(it+1, ITERS, fitness_list.min()))
    min_fitness_log.append(fitness_list.min().item())
    fitness_list -= fitness_list.mean()

    h = models.batched_entropy(distribution)

    optimizer.zero_grad()  # clear gradient buffers

    loss = ((logp * fitness_list).mean(1) - C * h).mean()

    loss.backward()  # update gradient buffers
    optimizer.step()  # update model's parameters

plt.plot(range(ITERS), min_fitness_log, label='min fitness')
plt.show()
