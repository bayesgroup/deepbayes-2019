import os
import sys
import random
import numpy as np

from collections import OrderedDict
from tabulate import tabulate
from pandas import DataFrame
from time import gmtime, strftime


class Logger:
    def __init__(self, name='name', fmt=None):
        self.handler = True
        self.scalar_metrics = OrderedDict()
        self.fmt = fmt if fmt else dict()

        base = './logs'
        if not os.path.exists(base): os.mkdir(base)

        time = gmtime()
        hash = ''.join([chr(random.randint(97, 122)) for _ in range(3)])
        fname = '-'.join(sys.argv[0].split('/')[-3:])
        self.path = '%s/%s-%s-%s-%s' % (base, fname, name, hash, strftime('%m-%d-%H:%M', time))

        self.logs = self.path + '.csv'
        self.output = self.path + '.out'
        self.checkpoint = self.path + '.cpt'

        def prin(*args):
            str_to_write = ' '.join(map(str, args))
            with open(self.output, 'a') as f:
                f.write(str_to_write + '\n')
                f.flush()

            print(str_to_write)
            sys.stdout.flush()

        self.print = prin

    def add_scalar(self, t, key, value):
        if key not in self.scalar_metrics:
            self.scalar_metrics[key] = []
        self.scalar_metrics[key] += [(t, value)]

    def iter_info(self, order=None):
        names = list(self.scalar_metrics.keys())
        if order:
            names = order
        values = [self.scalar_metrics[name][-1][1] for name in names]
        t = int(np.max([self.scalar_metrics[name][-1][0] for name in names]))
        fmt = ['%s'] + [self.fmt[name] if name in self.fmt else '.1f' for name in names]

        if self.handler:
            self.handler = False
            self.print(tabulate([[t] + values], ['epoch'] + names, floatfmt=fmt))
        else:
            self.print(tabulate([[t] + values], ['epoch'] + names, tablefmt='plain', floatfmt=fmt).split('\n')[1])

    def save(self):
        result = None
        for key in self.scalar_metrics.keys():
            if result is None:
                result = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
            else:
                df = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
                result = result.join(df, how='outer')
        result.to_csv(self.logs)
        self.print('The log/output/model have been saved to: ' + self.path + ' + .csv/.out/.cpt')