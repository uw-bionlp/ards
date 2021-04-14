
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict, Counter
import os
import numpy as np


BATCH = 'batch'
EPOCH = 'epoch'
TOTAL = 'total'
SEP = 'sep'


def tensor_check(x):
    if torch.is_tensor(x):
        x = x.tolist()
    return x

def movingaverage(y, N):

    y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid')
    return y_smooth

class PlotLoss():

    def __init__(self, path=None, window=5, max_len=1000, logplot=True):

        self.path = path
        self.window = window
        self.max_len = max_len
        self.logplot = logplot

        self.loss = OrderedDict()
        self.loss[BATCH] = OrderedDict()
        self.loss[BATCH][TOTAL] = []
        self.loss[BATCH][SEP] = OrderedDict()

        self.loss[EPOCH] = OrderedDict()
        self.loss[EPOCH][TOTAL] = []
        self.loss[EPOCH][SEP] = OrderedDict()


    def update(self, type_, loss, loss_dict=None):

        # do nothing if no path
        if self.path is None:
            return None



        loss_total = self.loss[type_][TOTAL]
        loss_sep = self.loss[type_][SEP]

        loss_total.append(tensor_check(loss))

        for k, v in loss_dict.items():
            if k not in loss_sep:
                loss_sep[k] = []
            loss_sep[k].append(tensor_check(v))

        #if len(loss_total) > self.max_len:
        #    X = range(0, len(loss_total), 10)
        #    self.loss[type_][TOTAL] = [loss_total[x] for x in X]
        #
        #    for k, v in loss_sep.items():
        #        self.loss[type_][SEP] = [v[x] for x in X]

    def plot(self, type_):

        # do nothing if no path
        if self.path is None:
            return None

        # create figure
        fig, ax = plt.subplots(1)

        loss_total = self.loss[type_][TOTAL]
        loss_sep = self.loss[type_][SEP]

        x = range(0, len(loss_total))

        for k, v in loss_sep.items():
            ax.plot(x, loss_sep[k], label=k)

        ax.plot(x, loss_total, label='total')
        smooth_loss = movingaverage(loss_total, self.window)

        if self.logplot:
            ax.semilogy(x, smooth_loss, label='smooth')
        else:
            ax.plot(x, smooth_loss, label='smooth')

        ax.set_ylabel('loss')
        ax.set_xlabel(type_)
        ax.legend(loc="lower left")

        f = os.path.join(self.path, f"loss_curve_{type_}.png")
        fig.savefig(f)
        plt.close(fig=fig)
        plt.close('all')
        return True


    def update_batch(self, loss, loss_dict=None):
        self.update(BATCH, loss, loss_dict=loss_dict)
        return True

    def update_epoch(self, loss, loss_dict=None):
        self.update(EPOCH, loss, loss_dict=loss_dict)
        self.plot(EPOCH)
        self.plot(BATCH)
        return True
