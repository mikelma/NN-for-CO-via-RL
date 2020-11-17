import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import csv


class DataLogger():
    def __init__(self, constants=None):
        self.log = {}
        self.constants = constants

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

    def plot(self, update=False):
        # TODO: Generalize this function
        if update:
            plt.clf()

        gs = gridspec.GridSpec(2, 2)

        # fitness related plots
        plt.subplot(gs[0, 0])
        self.plot_key('mean fitness')
        self.plot_key('min fitness')
        self.plot_key('best fitness')
        plt.title('Fitness of the sampled solutions')
        plt.xlabel('Iterations')
        plt.xlabel('Fitness value')
        plt.legend()

        # loss value related plots
        plt.subplot(gs[0, 1])
        self.plot_key('loss')
        plt.title('Loss value')
        plt.xlabel('Iterations')
        plt.xlabel('Loss')
        plt.legend()

        # model's convergency related plots
        # entopy
        ax1 = plt.subplot(gs[1, :])
        ax1.plot(range(len(self.log['entropy'])),
                 self.log['entropy'], color='tab:purple', label='entropy', alpha=.7)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Entopy')
        ax1.set_title("Model's convergency metrics")
        ax1.legend(loc=1)
        # instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()
        # plot prob. of samplig the best and worst solutions
        ax2.set_ylabel('Probability')
        key = 'best sol. prob.'
        ax2.plot(range(len(self.log[key])),
                 self.log[key], color='tab:orange', label=key, alpha=.7)
        key = 'worst sol. prob.'
        ax2.plot(range(len(self.log[key])),
                 self.log[key], 'b-', color='tab:blue', label=key, alpha=.7)
        ax2.legend(loc=4)

        if not update:
            plt.show()
        else:
            plt.pause(.0001)

    def to_csv(self, filename, num_rows):
        import csv
        import os

        write_header = not os.path.isfile(filename)
        with open(filename, 'a') as csvfile:
            writer = csv.writer(csvfile)

            # only write the hader if the logger file does not already exist
            if write_header:
                c = [] if self.constants == None else [k for k in self.constants.keys()]
                writer.writerow(c + [k for k in self.log.keys()])

            # write rows
            for i in range(num_rows):
                # write constants
                if self.constants == None:
                    c = []
                else:
                    c = [str(value)
                         for value in self.constants.values()]
                # write log's values
                d = [str(self.log[key][i]) for key in self.log.keys()]
                writer.writerow(c + d)


if __name__ == '__main__':
    dl = DataLogger()
    print(dl.log)
