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
LOG_FILE = None
INSTANCE = '../../instances/PFSP/tai20_5_8.fsp'
LOSS_FUNC = 'a2c'
BATCH_SIZE = 3


dl = DataLogger({'instance': INSTANCE.split('/')[-1],
                 'instance size': N,
                 'max iters': ITERS,
                 'num. samples': N_SAMPLES,
                 'learning rate': LR,
                 'noise length': NOISE_LEN,
                 'loss function': LOSS_FUNC,
                 })

problem = pypermu.problems.pfsp.Pfsp('../../instances/PFSP/tai20_5_8.fsp')

model = models.MultiHeadModel(NOISE_LEN, N, device=DEVICE)
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=LR)
value_loss = torch.nn.MSELoss(reduction='none')

for it in range(ITERS):
    # feed the model and get the output distribution
    noise = torch.rand((BATCH_SIZE, NOISE_LEN)).to(DEVICE)
    policy, values = model.get_distribution_and_value(noise)
    samples = models.sample_batched(policy, N_SAMPLES)

    permus = [[utils.marina2permu(v) for v in batch.cpu()
               ] for batch in samples]
    fitness_list = torch.tensor(
        [problem.evaluate(batch) for batch in permus]).float().to(DEVICE)

    optimizer.zero_grad()  # clear gradient buffers

    # compute loss
    target_values = fitness_list.mean(1).view(-1, 1)
    critic_loss = value_loss(values, target_values)
    logp = models.log_probs(samples, policy)
    print(logp)
    print(critic_loss)
    quit()

    loss = (logp * critic_loss).mean()

    loss.backward()  # update gradient buffers
    optimizer.step()  # update model's parameters

    # --------------------- logger --------------------- #
    with torch.no_grad():
        h = utils.entropy(policy)
    dl.push(fitness_list=fitness_list.cpu().numpy())
    dl.push(other={'iteration': it,
                   'entropy': h.item(),
                   'loss': loss.item(),
                   })
    print(it+1, '/', ITERS, end=' ')
    dl.print()

if LOG_FILE != None:
    dl.to_csv(LOG_FILE, ITERS)
# dl.plot()
