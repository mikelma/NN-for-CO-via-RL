import pypermu
import torch
from torch.optim import Adam
import numpy as np
import utils
import loss_funcs
import models
import os
import uuid

# --------------------- configuration --------------------- #
WANDB_NAME = 't1-batch-exp'
INST_PATH, INST_SIZE, WRITE_LOG, WANDB_ENABLE = utils.arg_parse()
DEVICE = 'cpu'

config = {'instance': INST_PATH.split('/')[-1],
          'instance size': INST_SIZE,
          'max iters': 1000,
          'n samples': 64,
          'learning rate': .0003,
          'noise length': 128,
          'loss function': 'L1',
          'eval inverse': True,
          'model': models.SimpleModel,
          'batch size': 1,  # NOTE: here batch size is always 1
          }
# --------------------------------------------------------- #

if WRITE_LOG:
    from datalog import DataLogger
    dl = DataLogger(config)

if WANDB_ENABLE:
    import wandb
    wandb.init(WANDB_NAME, config=config)

# initializations
problem = pypermu.problems.pfsp.Pfsp(INST_PATH)

model = config['model'](config['noise length'],
                        config['instance size'], device=DEVICE)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=config['learning rate'])

if WANDB_ENABLE:
    wandb.watch(model, log='all', log_freq=10)

for it in range(config['max iters']):
    # forward pass
    noise = torch.rand(config['noise length']).to(DEVICE)
    distribution = model.get_distribution(noise)
    samples = models.sample(distribution, config['n samples'])

    # convert the sampled marina vectors to their permutation representations
    permus = pypermu.utils.transformations.marina2permu_batched(
        samples.cpu().numpy())

    if config['eval inverse']:
        # transform the permus list into a list of it's inverse permutations
        permus = pypermu.utils.transformations.permu2inverse_batched(permus)

    # evaluate the fitness of the sampled solutions
    fitness_list = torch.as_tensor(
        problem.evaluate(permus)).float().to(DEVICE)

    # --------------------- logger --------------------- #
    if WANDB_ENABLE:
        wandb.log({
            'min fitness': fitness_list.min().item(),
            'mean fitness': fitness_list.mean().item(),
        }, step=it)

    if WRITE_LOG:
        dl.push(fitness_list=fitness_list.cpu().numpy())
    # -------------------------------------------------- #

    if config['loss function'] == 'L1':
        loss = loss_funcs.compute_l1(samples, distribution,
                                     fitness_list)
    # if config['loss function'] == 'L2':
    #    loss = loss_funcs.compute_l2(
    #        samples, distribution, fitness_list, config['C'])

    # elif config['loss function'] == 'L3':
    #    loss = loss_funcs.compute_l3(samples, distribution,
    #                                 fitness_list, N=len(distribution))
    # elif config['loss function'] == 'L4':
    #    loss = loss_funcs.compute_l4(
    #        samples, distribution, fitness_list)

    # elif config['loss function'] == 'L5':
    #    loss, convergency, scaled_logps = loss_funcs.compute_l5(
    #        samples, distribution, fitness_list, gamma=config['gamma'])

    optimizer.zero_grad()  # clear gradient buffers
    loss.backward()  # update gradient buffers
    optimizer.step()  # update model's parameters

    # --------------------- logger --------------------- #
    if WRITE_LOG or WANDB_ENABLE:
        with torch.no_grad():

            print('Recording entropies')
            h = utils.entropy(distribution, reduction='none')
            entropies = {}
            for i in range(config['instance size']):
                entropies['h'+str(i)] = h[i].item()

            data = {'entropy': h.sum().item(),
                    'loss': loss.item()}
            merged = {**entropies, **data}

            if WRITE_LOG:
                dl.push(other=merged)
            if WANDB_ENABLE:
                # print('distrib array recording')
                # m = np.zeros((N, N))
                # for permu in permus:
                #     for i, v in enumerate(permu):
                #         m[i][v] += 1
                # # np.save('arrays/{}'.format(it), m)
                # wandb.log({'permu distrib': wandb.Image(m)}, step=it)
                wandb.log(merged, step=it)
    else:
        print(it+1, '/', config['max iters'])

if WANDB_ENABLE:
    torch.onnx.export(model, noise, "model.onnx")
    wandb.save("model.onnx")

if WRITE_LOG:
    dl.to_csv('runs/'+str(uuid.uuid4())+'.csv', config['max iters'])
