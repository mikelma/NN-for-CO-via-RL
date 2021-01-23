import models
import torch
import utils
from torch.optim import Adam
from datalog import DataLogger
import uuid
import pypermu


# --------------------- configuration --------------------- #
WANDB_NAME = 't1-batch-exp'
INST_PATH, INST_SIZE, WRITE_LOG, WANDB_ENABLE = utils.arg_parse()
DEVICE = 'cpu'
config = {'instance': INST_PATH.split('/')[-1],
          'instance size': INST_SIZE,
          'max iters': 1000,
          'n samples': 64,
          'learning rate': .00003,
          'noise length': 128,
          'loss function': 'L1',
          'eval inverse': True,
          'model': models.SimpleModelBatched,
          'batch size': 128,
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

for it in range(config['max iters']):
    noise = torch.rand(
        (config['batch size'], config['noise length'])).to(DEVICE)

    distribution, samples, logp = model.get_samples_and_logp(
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
        wandb.log({
            'min fitness': fitness_list.min().item(),
            'mean fitness': fitness_list.mean().item(),
        }, step=it)

    if WRITE_LOG:
        dl.push(fitness_list=fitness_list.cpu().numpy())
    # -------------------------------------------------- #

    if config['loss function'] == 'L1':
        # utility function (operation is done per batch)
        fitness_list -= fitness_list.mean(1).view(-1, 1)
        # compute final loss
        loss = (logp * fitness_list).mean()

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
    dl.to_csv('runs/'+str(uuid.uuid4())+'.csv', config['max iters'])
