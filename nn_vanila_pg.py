import pypermu
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
from datalog import DataLogger
import utils
import loss_funcs
import models

DEVICE = 'cuda:0'
NOISE_LEN = 128
N = 20
N_SAMPLES = 64
LR = .0003
ITERS = 1000
C = 40
LOG_FILE = None
LOSS_FUNC = 'L3'
INSTANCE = '../../instances/PFSP/tai20_5_8.fsp'


dl = DataLogger({'instance': INSTANCE.split('/')[-1],
                 'instance size': N,
                 'max iters': ITERS,
                 'num. samples': N_SAMPLES,
                 'learning rate': LR,
                 'noise length': NOISE_LEN,
                 'C': C,
                 'loss function': LOSS_FUNC,
                 })

problem = pypermu.problems.pfsp.Pfsp('../../instances/PFSP/tai20_5_8.fsp')

model = models.SimpleModel(NOISE_LEN, N, device=DEVICE)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

for it in range(ITERS):
    noise = torch.rand(NOISE_LEN).to(DEVICE)
    distribution = model.get_distribution(noise)
    samples = models.sample(distribution, N_SAMPLES)

    permus = [utils.marina2permu(v) for v in samples.cpu()]
    fitness_list = torch.tensor(
        problem.evaluate(permus)).float().to(DEVICE)

    dl.push(fitness_list=fitness_list.cpu().numpy())

    fitness_list -= fitness_list.mean()

    h = utils.entropy(distribution)

    optimizer.zero_grad()  # clear gradient buffers

    if LOSS_FUNC == 'L2':
        loss = loss_funcs.compute_l2(
            samples, distribution, fitness_list, h, c=C)
    elif LOSS_FUNC == 'L3':
        loss = loss_funcs.compute_l3(samples, distribution,
                                     fitness_list, h, N=len(distribution))

    loss.backward()  # update gradient buffers
    optimizer.step()  # update model's parameters

    # --------------------- logger --------------------- #
    dl.push(other={'iteration': it,
                   'entropy': h.item(),
                   'loss': loss.item(),
                   })
    print(it+1, '/', ITERS, end=' ')
    dl.print()

if LOG_FILE != None:
    dl.to_csv(LOG_FILE, ITERS)
# dl.plot()
