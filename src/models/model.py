
import os
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
#from tensorboardX import SummaryWriter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter, OrderedDict
import random


import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import joblib
import math
import os
from utils.proj_setup import make_and_clear
from config.constants import PARAMS_FILE, STATE_DICT, PREDICTIONS_FILE
from layers.utils import set_model_device, set_tensor_device


TUNE = 'tune'
PREDICT = 'predict'



class Model(nn.Module):

    '''
    Base model
    '''

    def __init__(self,  \
        hyperparams,
        dataset_params,
        dataloader_params,
        optimizer_params,
        num_workers,
        num_epochs,
        dataset_class,
        scorer_class
        ):

        self.hyperparams = hyperparams
        self.dataset_params = dataset_params
        self.dataloader_params = dataloader_params
        self.optimizer_params = optimizer_params
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.dataset_class = dataset_class
        self.scorer = scorer_class()

        super(Model, self).__init__()


        self.params = OrderedDict()
        self.params['hyperparams'] = hyperparams
        self.params['dataset_params'] = dataset_params
        self.params['dataloader_params'] = dataloader_params
        self.params['optimizer_params'] = optimizer_params
        self.params['num_workers'] = num_workers
        self.params['num_epochs'] = num_epochs
        self.params['scorer_class'] = scorer_class


    def reset_parameters(self):


        i = 0
        parameter_count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.split('.')[-1] == 'bias':
                    torch.nn.init.zeros_(param.data)
                else:
                    torch.nn.init.normal_(param.data)

                parameter_count += param.numel()

                i += 1
        logging.info(f"Reset {i} model variable")
        logging.info(f"Reset {parameter_count} model parameters")
        return True


    def foward(self, X, y):
        NotImplementedError("no forward method implemented")

    def fit(self, X, y, device=None, path=None, shuffle=True):
        NotImplementedError("no fit method implemented")

    def predict(self, X, device=None, path=None):
        NotImplementedError("no predict method implemented")

    def prob(self, X, device=None, path=None):
        NotImplementedError("no prob method implemented")


    def score(self, X, y, device=None, path=None):
        '''
        Generate predictions and score result
        '''

        # get predictions
        y_pred = self.predict(X, device=device, path=path)

        # score predictions
        scores = self.scorer.fit(y, y_pred, path=path)

        return (y_pred, scores)


    def fit_cv(self, X, y, device=None, path=None, n_splits=3, shuffle=True, seed=1):

        if shuffle:
            z = list(zip(X, y))
            random.Random(seed).shuffle(z)
            X, y = zip(*z)
            if not isinstance(X, list):
                X = list(X)
            if not isinstance(y, list):
                y = list(y)


        kf = KFold(n_splits=n_splits)


        dfs = OrderedDict()
        for j, (train_index, test_index) in enumerate(kf.split(X)):

            self.reset_parameters()

            X_train = [X[i] for i in train_index]
            y_train = [y[i] for i in train_index]

            X_test = [X[i] for i in test_index]
            y_test = [y[i] for i in test_index]

            dir = os.path.join(path, f'cross_val_{j}')
            make_and_clear(dir)

            self.fit(X_train, y_train, device=device, path=dir)
            y_pred, scores = self.score(X_test, y_test, device=device, path=dir)

            for name, df in scores.items():
                if name not in dfs:
                    dfs[name] = []
                dfs[name].append(df)

        dfs = self.scorer.combine_cv(dfs, path=path)

        return dfs

    def save(self, path):
        '''
        Save model
        '''


        # Save state dict
        state_dict = self.state_dict()

        if hasattr(self, 'state_dict_exclusions'):
            for excl in self.state_dict_exclusions:
                if excl in state_dict:
                    del state_dict[excl]
        f = os.path.join(path, STATE_DICT)
        torch.save(state_dict, f)

        # save parameter file
        f = os.path.join(path, PARAMS_FILE)
        joblib.dump(self.params, f)

        return True

    def get_summary(self):
        '''
        Generate and print summary of model
        '''

        # Print model summary
        #logging.info("\n")
        #logging.info("Model summary")
        #logging.info(self)

        # Print trainable parameters
        logging.info("\n")
        logging.info("Trainable parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                logging.info('\t{}\t{}'.format(name, param.size()))

        logging.info("\n")
        num_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_pM = num_p/1e6
        logging.info("Total trainable parameters:\t{:.1f} M".format(num_pM))
        logging.info("Total trainable parameters:\t{}".format(num_p))
        logging.info("\n")
