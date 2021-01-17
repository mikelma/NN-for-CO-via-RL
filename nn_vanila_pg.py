import pypermu
import torch
from torch.optim import Adam
import numpy as np
import utils
import loss_funcs
import models
import os
import uuid
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--instance-size', metavar='N', type=int, nargs=1,
                    required=True, help='Size of the instance', choices=[20, 50])
parser.add_argument('--instance', metavar='FILE', type=str, nargs=1,
                    required=True, help='Path to the instance file')
parser.add_argument('--log', type=str, nargs='?', default=False, const=True,
                    required=False, help='If this falg is set, the logger will be stored as a CSV')
parser.add_argument('--wandb', type=str, nargs='?', default=False, const=True,
                    required=False, help='If this falg is provided, weight and biases will be used to track the experiment')
args = parser.parse_args()

WRITE_LOG = False if args.log == False else True
WANDB_ENABLE = False if args.wandb == False else True
DEVICE = 'cpu'
instance_path = args.instance[0]

config = {'instance': instance_path.split('/')[-1],
          'instance size': args.instance_size[0],
          'max iters': 1000,
          'n samples': 64,
          'learning rate': .0003,
          'noise length': 128,
          'C': 40,
          'loss function': 'L6',
          'eval inverse': True,
          'borda': False,
          'model': models.SimpleModel,
          'gamma': .02,
          }

if WRITE_LOG:
    from datalog import DataLogger
    dl = DataLogger(config)

if WANDB_ENABLE:
    import wandb
    wandb.init('NN-PG-loss-test', config=config)

# initializations
problem = pypermu.problems.pfsp.Pfsp(instance_path)

model = config['model'](config['noise length'],
                        config['instance size'], device=DEVICE)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=config['learning rate'])

if WANDB_ENABLE:
    wandb.watch(model, loss_funcs.compute_l6, log='all', log_freq=10)

for it in range(config['max iters']):
    # forward pass
    noise = torch.rand(config['noise length']).to(DEVICE)
    distribution = model.get_distribution(noise)
    samples = models.sample(distribution, config['n samples'])

    # convert the sampled marina vectors to their permutation representations
    permus = pypermu.utils.transformations.marina2permu_batched(
        samples.cpu().numpy())

    if config['borda']:
        borda = np.array(pypermu.utils.borda(permus))
        inv_borda = utils.permu2inverse(borda)
        permus = [np.array(pypermu.utils.compose(p, inv_borda))
                  for p in permus]

    if config['eval inverse']:
        # transform the permus list into a list of it's inverse permutations
        permus = pypermu.utils.transformations.permu2inverse_batched(permus)

    # evaluate the fitness of the sampled solutions
    fitness_list = torch.as_tensor(
        problem.evaluate(permus)).float().to(DEVICE)

    #################################################
    if WANDB_ENABLE:
        wandb.log({
            'min fitness': fitness_list.min().item(),
            'mean fitness': fitness_list.mean().item(),
        }, step=it)

    if WRITE_LOG:
        dl.push(fitness_list=fitness_list.cpu().numpy())
    #################################################

    optimizer.zero_grad()  # clear gradient buffers

    if config['loss function'] == 'L2':
        loss = loss_funcs.compute_l2(
            samples, distribution, fitness_list, config['C'])

    elif config['loss function'] == 'L3':
        loss = loss_funcs.compute_l3(samples, distribution,
                                     fitness_list, N=len(distribution))
    elif config['loss function'] == 'L4':
        loss = loss_funcs.compute_l4(
            samples, distribution, fitness_list)

    elif config['loss function'] == 'L5':
        loss, convergency, scaled_logps = loss_funcs.compute_l5(
            samples, distribution, fitness_list, gamma=config['gamma'])

    elif config['loss function'] == 'L6':
        loss, convergency = loss_funcs.compute_l6(samples, distribution,
                                                  fitness_list)

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

            data = {'iteration': it, 'entropy': h.sum().item(
            ), 'loss': loss.item(), 'convergency': convergency.item()}
            merged = {**entropies, **data}

            if WRITE_LOG:
                dl.push(other=merged)
            if WANDB_ENABLE:
                ###########################
                # print('distrib array recording')
                # m = np.zeros((N, N))
                # for permu in permus:
                #     for i, v in enumerate(permu):
                #         m[i][v] += 1
                # # np.save('arrays/{}'.format(it), m)
                # wandb.log({'permu distrib': wandb.Image(m)}, step=it)
                ###########################
                wandb.log(merged, step=it)
    else:
        print(it+1, '/', config['max iters'])

if WANDB_ENABLE:
    torch.onnx.export(model, noise, "model.onnx")
    wandb.save("model.onnx")

if WRITE_LOG:
    dl.to_csv('runs/'+str(uuid.uuid4())+'.csv', config['max iters'])
