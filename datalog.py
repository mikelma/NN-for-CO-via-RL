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

    def print(self, ignore=[]):
        for key in self.log.keys():
            if self.log[key] and key not in ignore:
                print((key+': {:10.3f}').format(self.log[key][-1]), end=' ')
        print()

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
