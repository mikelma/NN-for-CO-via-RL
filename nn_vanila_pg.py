import matplotlib.pyplot as plt
import pypermu
import torch
from torch.optim import Adam
import numpy as np
from datalog import DataLogger
import utils
import loss_funcs
import models

NOISE_LEN = 128
N_SAMPLES = 64
LR = .0003
ITERS = 4000
LOSS_FUNC = 'L3'
C = 40
EVAL_INVERSE = True
BORDA = False

DEVICE = 'cpu'
LOG_FILE = None
N = 20
INSTANCE = '../../instances/PFSP/tai20_5_8.fsp'


dl = DataLogger({'instance': INSTANCE.split('/')[-1],
                 'instance size': N,
                 'max iters': ITERS,
                 'num. samples': N_SAMPLES,
                 'learning rate': LR,
                 'noise length': NOISE_LEN,
                 'C': C,
                 'loss function': LOSS_FUNC,
                 'eval inverse': EVAL_INVERSE,
                 })

problem = pypermu.problems.pfsp.Pfsp(INSTANCE)

model = models.SimpleModel(NOISE_LEN, N, device=DEVICE)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

for it in range(ITERS):
    print('DEVICE: ', DEVICE)
    # forward pass
    noise = torch.rand(NOISE_LEN).to(DEVICE)
    distribution = model.get_distribution(noise)
    samples = models.sample(distribution, N_SAMPLES)

    permus = [utils.marina2permu(v) for v in samples.cpu()]
    if BORDA:
        borda = np.array(pypermu.utils.borda(permus))
        inv_borda = utils.permu2inverse(borda)
        permus = [np.array(pypermu.utils.compose(p, inv_borda))
                  for p in permus]

    if EVAL_INVERSE:
        # transform the permus list into a list of it's inverse permutations
        permus = [utils.permu2inverse(permu) for permu in permus]

    # evaluate the fitness of the sampled solutions
    fitness_list = torch.tensor(
        problem.evaluate(permus)).float().to(DEVICE)

    dl.push(fitness_list=fitness_list.cpu().numpy())

    fitness_list -= fitness_list.mean()

    h = utils.entropy(distribution)

    optimizer.zero_grad()  # clear gradient buffers

    if LOSS_FUNC == 'L2':
        loss = loss_funcs.compute_l2(
            samples, distribution, fitness_list, h, C)
    elif LOSS_FUNC == 'L3':
        loss = loss_funcs.compute_l3(samples, distribution,
                                     fitness_list, h, N=len(distribution))

    loss.backward()  # update gradient buffers
    optimizer.step()  # update model's parameters

    # --------------------- logger --------------------- #
    with torch.no_grad():
        logp = models.log_probs_unprocessed(samples, distribution)
        h_list = utils.entropy(distribution, reduction='none')
        dl.push(other={'iteration': it,
                       'entropy': h.item(),
                       'loss': loss.item(),
                       })
        print(it+1, '/', ITERS, end=' ')
        dl.print()

if LOG_FILE != None:
    dl.to_csv(LOG_FILE, ITERS)

dl.plot()
