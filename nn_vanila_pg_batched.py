import models
import torch
import utils
from torch.optim import Adam
import uuid
import pypermu
from loss_funcs_batched import loss_l1, loss_l5, mean_utility, standardized_utility
import numpy as np
import os


import matplotlib.pyplot as plt


# ------------------------ params ------------------------- #
MODEL = models.SimpleModelBatched
BATCH_BOUND = 'upper'
# MODEL = models.TinyBatched
# BATCH_BOUND = 'lower'

LOSS = 'L1'
N_SAMPLES = 64

LOG_DIR = './'
WANDB_NAME = 'depth-comp'
# --------------------------------------------------------- #

# --------------------- configuration --------------------- #
INST_PATH, INST_SIZE, WRITE_LOG, WANDB_ENABLE = utils.arg_parse()

MAX_ITERS, BATCH_SIZE = utils.get_max_iters_and_batch_size(
    INST_SIZE, N_SAMPLES, 1000*INST_SIZE**2, batch_size_bound=BATCH_BOUND)

# choose device depending if the script is run in my host machine
# or remotely in the cluster
DEVICE = 'cuda:0' if os.uname()[1] == 'marvin' else 'cpu'

config = {'instance': INST_PATH.split('/')[-1],
          'instance size': INST_SIZE,
          'max iters': MAX_ITERS,
          'n samples': N_SAMPLES,
          'learning rate': .0003,
          'noise length': 128,
          'loss function': LOSS,
          'eval inverse': True,
          'model': MODEL.name,
          'batch size': BATCH_SIZE,
          'gamma': 1,
          'utility': mean_utility,
          }

if WANDB_ENABLE:
    import wandb
    wandb.init(WANDB_NAME, config=config)
# --------------------------------------------------------- #

problem = pypermu.problems.pfsp.Pfsp(INST_PATH)
model = MODEL(
    config['noise length'], config['instance size'], device=DEVICE)
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=config['learning rate'])

fig, axes = plt.subplots(1, 3)

best_fitness = float('inf')
for it in range(config['max iters']):
    noise = torch.rand(
        (config['batch size'], config['noise length'])).to(DEVICE)

    distribution, samples, logps = model.get_samples_and_logp(
        noise, config['n samples'])

    # convert sampled marina vectors to their permutation representation
    permus = [pypermu.utils.transformations.marina2permu_population(
        b) for b in samples.cpu().numpy()]

    if config['eval inverse']:
        # calculate the inverse of the permutations
        permus = [pypermu.utils.transformations.permu2inverse_population(
            batch) for batch in permus]

    # evaluate the inversed permutations
    fitness_list = torch.as_tensor([problem.evaluate(batch)
                                    for batch in permus]).float().to(DEVICE)

    # --------------------- logger --------------------- #
    if WRITE_LOG or WANDB_ENABLE:
        min_f = fitness_list.min().item()
        best_fitness = best_fitness if min_f >= best_fitness else min_f

        if WANDB_ENABLE:
            wandb.log({
                'min fitness': min_f,
                'mean fitness': fitness_list.mean().item(),
                'best fitness': best_fitness,
            }, step=it)
    # -------------------------------------------------- #

    if config['loss function'] == 'L1':
        loss = loss_l1(fitness_list, logps, utility=config['utility'])

    elif config['loss function'] == 'L5':
        loss = loss_l5(fitness_list, logps, distribution,
                       config['gamma'], utility=config['utility'], device=DEVICE)

    optimizer.zero_grad()  # clear gradient buffers
    loss.backward()  # update gradient buffers

    # ---------------------------------------- #
    if True:
        g = model.l1.weight.detach().cpu().numpy()
        mean_g_1 = np.absolute(g).mean()

        g = model.l1.bias.detach().cpu().numpy()
        mean_b_1 = np.absolute(g).mean()

        # axes[0].cla()
        # p = axes[0].matshow(g)
        # b1 = fig.colorbar(p, ax=axes[0])
        # axes[0].set_title(f'Shared layer')

        g = model.out_layers[0].weight.detach().cpu().numpy().T
        mean_g_2 = np.absolute(g).mean()

        g = model.out_layers[0].bias.detach().cpu().numpy()
        mean_b_2 = np.absolute(g).mean()
        # axes[1].cla()
        # p = axes[1].matshow(g)
        # b2 = fig.colorbar(p, ax=axes[1])
        # axes[1].set_title('Out0')

        g = model.out_layers[9].weight.detach().cpu().numpy().T
        mean_g_3 = np.absolute(g).mean()

        g = model.out_layers[9].bias.detach().cpu().numpy()
        mean_b_3 = np.absolute(g).mean()
        # axes[2].cla()
        # p = axes[2].matshow(g)
        # b3 = fig.colorbar(p, ax=axes[2])
        # axes[2].set_title('Out9')

        # plt.savefig('imgs/{:0>3}.png'.format(it))

        # wandb.log({'weigth values': [wandb.Image(
        #     plt, caption='Weigth matrix values from different layers')]}, step=it)

        wandb.log({'weigth magnitude shared1': mean_g_1,
                   'weigth magnitude out0': mean_g_2,
                   'weigth magnitude out9': mean_g_3,
                   'bias magnitude shared1': mean_b_1,
                   'bias magnitude out0': mean_b_2,
                   'bias magnitude out9': mean_b_3,
                   }, step=it)

        # wandb.log({'shared layer out': [wandb.Histogram(
        #     shared_out.detach().cpu().numpy())]}, step=it)

        # b1.remove()
        # b2.remove()
        # b3.remove()
        # for i in [0, 4, 9, 17]:
        #     g = model.layers[9].weight.grad.cpu().numpy().T
        #     mean_grad = np.absolute(g).mean()
        #     wandb.log({'grad mean out'+str(i): mean_grad}, step=it)

        # ---------------------------------------- #
    optimizer.step()  # update model's parameters

    if WANDB_ENABLE:
        with torch.no_grad():
            # entropy of all N distributions (averaged across batches)
            # h = models.batched_entropies(distribution).mean(0)
            # entropies = {}
            # for i in range(config['instance size']):
            #     entropies['h'+str(i)] = h[i].item()

            # data = {'entropy': h.sum().item(),
            #         'loss': loss.item()}
            # merged = {**entropies, **data}
            merged = {'loss': loss.item()}

            if WANDB_ENABLE:
                wandb.log(merged, step=it)

# if WANDB_ENABLE:
#    torch.onnx.export(model, noise, "net.onnx")
#    wandb.save("net.onnx")

if WRITE_LOG:
    with open(str(uuid.uuid4())+'.csv', 'w') as f:
        f.writelines([
            ','.join([key for key in config.keys()]+['best fitness'])+'\n',
            ','.join([str(config[key])
                      for key in config.keys()]+[str(best_fitness)])+'\n'])
