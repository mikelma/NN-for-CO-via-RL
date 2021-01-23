import models
import torch
import utils
from torch.optim import Adam
from datalog import DataLogger
import uuid
import pypermu
from loss_funcs_batched import loss_l1, loss_l5


# --------------------- configuration --------------------- #
LOG_DIR = './'
WANDB_NAME = 't2-l5-gamma'
INST_PATH, INST_SIZE, WRITE_LOG, WANDB_ENABLE = utils.arg_parse()
DEVICE = 'cpu'
config = {'instance': INST_PATH.split('/')[-1],
          'instance size': INST_SIZE,
          'max iters': 1000,
          'n samples': 64,
          'learning rate': .0003,
          'noise length': 128,
          'loss function': 'L5',
          'eval inverse': True,
          'model': models.SimpleModelBatched,
          'batch size': 1,
          'gamma': 1,
          }
if WRITE_LOG:
    from datalog import DataLogger
    dl = DataLogger(config)

if WANDB_ENABLE:
    import wandb
    wandb.init(WANDB_NAME, config=config)
# --------------------------------------------------------- #

problem = pypermu.problems.pfsp.Pfsp(INST_PATH)
model = config['model'](
    config['noise length'], config['instance size'], device=DEVICE)
model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=config['learning rate'])

best_fitness = float('inf')

for it in range(config['max iters']):
    noise = torch.rand(
        (config['batch size'], config['noise length'])).to(DEVICE)

    distribution, samples, logps = model.get_samples_and_logp(
        noise, config['n samples'])

    # convert sampled marina vectors to their permutation representation
    permus = [pypermu.utils.transformations.marina2permu_batched(
        b) for b in samples.cpu().numpy()]

    if config['eval inverse']:
        # calculate the inverse of the permutations
        permus = [pypermu.utils.transformations.permu2inverse_batched(
            batch) for batch in permus]

    # evaluate the inversed permutations
    fitness_list = torch.as_tensor([problem.evaluate(batch)
                                    for batch in permus]).float().to(DEVICE)

    # --------------------- logger --------------------- #
    if WANDB_ENABLE:
        min_f = fitness_list.min().item()
        best_fitness = best_fitness if min_f >= best_fitness else min_f
        wandb.log({
            'min fitness': min_f,
            'mean fitness': fitness_list.mean().item(),
            'best fitness': best_fitness,
        }, step=it)

    if WRITE_LOG:
        dl.push(fitness_list=fitness_list.cpu().numpy())
    # -------------------------------------------------- #

    if config['loss function'] == 'L1':
        loss = loss_l1(fitness_list, logps)

    elif config['loss function'] == 'L5':
        loss = loss_l5(fitness_list, logps, distribution, config['gamma'])

    optimizer.zero_grad()  # clear gradient buffers
    loss.backward()  # update gradient buffers
    optimizer.step()  # update model's parameters

    if WRITE_LOG or WANDB_ENABLE:
        with torch.no_grad():
            # entropy of all N distributions (averaged across batches)
            h = models.batched_entropies(distribution).mean(0)
            entropies = {}
            for i in range(config['instance size']):
                entropies['h'+str(i)] = h[i].item()

            data = {'entropy': h.sum().item(),
                    'loss': loss.item()}
            merged = {**entropies, **data}

            if WRITE_LOG:
                dl.push(other=merged)
            if WANDB_ENABLE:
                wandb.log(merged, step=it)
    else:
        print(it+1, '/', config['max iters'])

if WANDB_ENABLE:
    torch.onnx.export(model, noise, "model.onnx")
    wandb.save("model.onnx")

if WRITE_LOG:
    dl.to_csv(LOG_DIR+str(uuid.uuid4())+'.csv', config['max iters'])
