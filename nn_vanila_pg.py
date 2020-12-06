import pypermu
import torch
from torch.optim import Adam
import numpy as np
from datalog import DataLogger
import utils
import loss_funcs
import models
import uuid
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--instance-size', metavar='N', type=int, nargs=1,
                    required=True, help='Size of the instance', choices=[20, 50])
parser.add_argument('--instance', metavar='FILE', type=str, nargs=1,
                    required=True, help='Path to the instance file')
parser.add_argument('--log', metavar='FILE', type=str, nargs='?', default=False, const=True,
                    required=False, help='If this falg is set, the logger will be stored as a CSV')
args = parser.parse_args()

N = args.instance_size[0]
INSTANCE = args.instance[0]
WRITE_LOG = False if args.log == False else True

##############################
#   Common hyperparameters   #
##############################
DEVICE = 'cpu'

LOSS_FUNC = 'L2'
C = 40
EVAL_INVERSE = True
##############################
##############################

#----------------------------#
#       params: N=20         #
#----------------------------#
if N == 20:
    NOISE_LEN = 128
    N_SAMPLES = 64
    LR = .0003
    ITERS = 1000
    MODEL = models.SimpleModel
#----------------------------#

#----------------------------#
#       params: N=50         #
#----------------------------#
if N == 50:
    NOISE_LEN = 128
    N_SAMPLES = 64
    LR = .0003
    ITERS = 1000
    MODEL = models.SimpleModel
#----------------------------#

dl = DataLogger({'instance': INSTANCE.split('/')[-1],
                 'instance size': N,
                 'max iters': ITERS,
                 'num. samples': N_SAMPLES,
                 'learning rate': LR,
                 'noise length': NOISE_LEN,
                 'C': C,
                 'loss function': LOSS_FUNC,
                 'eval inverse': EVAL_INVERSE,
                 'model': MODEL.name,
                 })

problem = pypermu.problems.pfsp.Pfsp(INSTANCE)
model = MODEL(NOISE_LEN, N, device=DEVICE)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)

for it in range(ITERS):
    # forward pass
    noise = torch.rand(NOISE_LEN).to(DEVICE)
    distribution = model.get_distribution(noise)
    samples = models.sample(distribution, N_SAMPLES)

    permus = [utils.marina2permu(v) for v in samples.cpu()]

    if EVAL_INVERSE:
        # transform the permus list into a list of it's inverse permutations
        permus = [utils.permu2inverse(permu) for permu in permus]

    # evaluate the fitness of the sampled solutions
    fitness_list = torch.tensor(
        problem.evaluate(permus)).float().to(DEVICE)

    dl.push(fitness_list=fitness_list.cpu().numpy())

    fitness_list -= fitness_list.mean()

    if LOSS_FUNC in ['L2', 'L3']:
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
        if not WRITE_LOG:
            print(it+1, '/', ITERS, end=' ')
            dl.print()

if WRITE_LOG:
    dl.to_csv(str(uuid.uuid4())+'.csv', ITERS)
