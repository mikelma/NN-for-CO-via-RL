import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class DataLogger():
    def __init__(self):
        self.log = {}

        self.init_logger()

    def init_logger(self):
        self.log['min fitness'] = []
        self.log['mean fitness'] = []
        self.log['best fitness'] = []

    def push(self, fitness_list=None, other=None):

        if fitness_list is not None:
            min_f = np.min(fitness_list)
            self.log['min fitness'].append(min_f)
            self.log['mean fitness'].append(np.mean(fitness_list))

            old_best_f = self.log['best fitness'][-1] if len(
                self.log['best fitness']) > 0 else np.inf
            self.log['best fitness'].append(
                min_f if min_f < old_best_f else old_best_f)

        if other != None:
            for key in other.keys():
                # if the key does no exist in the logger
                if key not in self.log:
                    self.log[key] = []  # init
                self.log[key].append(other[key])

    def print(self):
        for key in self.log.keys():
            if self.log[key]:
                print((key+': {:10.3f}').format(self.log[key][-1]), end=' ')
        print()

    def plot_key(self, key):
        plt.plot(range(len(self.log[key])),
                 self.log[key], label=key)
        plt.title(key)

    def plot(self, update=False):
        if update:
            plt.clf()

        gs = gridspec.GridSpec(2, 2)

        # fitness related plots
        plt.subplot(gs[0, 0])
        self.plot_key('mean fitness')
        self.plot_key('min fitness')
        self.plot_key('best fitness')
        plt.legend()

        # loss value related plots
        plt.subplot(gs[0, 1])
        self.plot_key('loss')
        plt.legend()

        plt.subplot(gs[1, :])
        self.plot_key('entropy')
        plt.legend()

        if not update:
            plt.show()
        else:
            plt.pause(.0001)


if __name__ == '__main__':
    dl = DataLogger()
    print(dl.log)
