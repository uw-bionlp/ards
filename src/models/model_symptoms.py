
from collections import Counter, OrderedDict
import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader

from allennlp.modules import FeedForward
from allennlp.nn.activations import Activation


import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import joblib
import math
import os



from models.model import Model, TUNE, PREDICT
from layers.attention import Attention
from layers.recurrent import RNN
from modules.relation_extractor import RelationExtractor
from modules.hierarchical_document_classifier import HierarchicalClassifierMulti
from layers.utils import set_model_device, set_tensor_device
from models.dataset_symptoms import DatasetSymptoms
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL
from config.constants import ENTITIES, RELATIONS, DOC_LABELS
from layers.utils import get_loss, aggregate, PRFAggregator
from layers.plotting import PlotLoss

from scoring.scorer_symptoms import ScorerSymptoms

def tensor_dict_collect(X):

    # Build list of dictionaries
    d = OrderedDict()

    for i, x in enumerate(X):

        if i == 0:
            for k in x:
                d[k] = []

        for k, v in x.items():
            d[k].append(v)

    for k, v in d.items():
        d[k] = torch.stack(v, dim=0)

    return d






class ModelSymptoms(Model):

    def __init__(self,  \
        hyperparams,
        dataset_params,
        dataloader_params,
        optimizer_params,
        num_workers,
        num_epochs,
        dataset_class = DatasetSymptoms,
        scorer_class = ScorerSymptoms
        ):

        super().__init__( \
            hyperparams = hyperparams,
            dataset_params = dataset_params,
            dataloader_params = dataloader_params,
            optimizer_params = optimizer_params,
            num_workers = num_workers,
            num_epochs = num_epochs,
            dataset_class = dataset_class,
            scorer_class = scorer_class
            )


        self.use_rnn = self.hyperparams['use_rnn']
        self.loss_reduction = self.hyperparams['loss_reduction']

        if self.use_rnn:
            self.rnn = RNN(**self.hyperparams['rnn'])

        self.relation_extractor = RelationExtractor(**self.hyperparams['relation_extractor'])

        self.get_summary()


    def forward(self, seq_tensor, seq_mask, span_indices, span_mask, verbose=False):

        # seq_tensor (batch_size, sentence_length, embed_dim)
        # seq_mask (batch_size, sentence_length)
        # span_indices (span_count, 2)
        # span_mask (batch_size, span_count)

        output = OrderedDict()

        # recurrent layer
        if self.use_rnn:
            seq_tensor = self.rnn(seq_tensor, seq_mask)

        # relation extraction
        span_scores, top_role_scores, top_span_mask, top_indices = \
                        self.relation_extractor( \
                                    seq_tensor = seq_tensor,
                                    span_indices = span_indices,
                                    span_mask = span_mask,
                                    seq_mask = seq_mask,
                                    verbose = verbose)

        output["span_scores"] = span_scores
        output["top_role_scores"] = top_role_scores
        output["top_span_mask"] = top_span_mask
        output["top_indices"] = top_indices

        return output


    # OVERRIDE
    def fit(self, X, y, device=None, path=None, shuffle=True):

        logging.info('')
        logging.info('='*72)
        logging.info("Fit")
        logging.info('='*72)

        # Get/set device
        set_model_device(self, device)

        # Configure training mode
        self.train()

        # Set number of cores
        torch.set_num_threads(self.num_workers)

        # Create data set
        dataset = self.dataset_class(X, y=y, **self.dataset_params, device=device)

        # Create data loader
        dataloader = DataLoader(dataset, shuffle=shuffle, **self.dataloader_params)

        # Create optimizer
        optimizer = optim.Adam(self.parameters(), **self.optimizer_params)

        # Create loss plotter
        plotter = PlotLoss(path=path)

        # Create prf aggregator
        prf_agg = PRFAggregator()

        # Loop on epochs
        pbar = tqdm(total=self.num_epochs)
        for j in range(self.num_epochs):

            loss_epoch = 0
            losses_epoch = OrderedDict()
            prf = []

            # Loop on mini-batches
            for i, (indices, seq_tensor, seq_mask, span_indices, span_mask, y_true) in enumerate(dataloader):

                verbose = False #(i == 0) and (j == 0)

                # Reset gradients
                self.zero_grad()

                y_pred = self( \
                                seq_tensor = seq_tensor,
                                seq_mask = seq_mask,
                                span_indices = span_indices,
                                span_mask = span_mask,
                                verbose = verbose)

                loss, loss_dict = self.loss(y_true, y_pred)

                plotter.update_batch(loss, loss_dict)

                prf_agg.update_counts(self.perf_counts(y_true, y_pred))

                # Backprop loss
                loss.backward()

                loss_epoch += loss.item()
                for k, v in loss_dict.items():
                    if i == 0:
                        losses_epoch[k] = v.item()
                    else:
                        losses_epoch[k] += v.item()

                # Clip loss
                clip_grad_norm_(self.parameters(), self.hyperparams['grad_max_norm'])

                # Update
                optimizer.step()

            plotter.update_epoch(loss_epoch, losses_epoch)

            msg = []
            msg.append('epoch={}'.format(j))
            msg.append('{}={:.1e}'.format('Total', loss_epoch))
            for k, ls in losses_epoch.items():
                msg.append('{}={:.1e}'.format(k, ls))

            msg.append(prf_agg.prf())
            prf_agg.reset()

            msg = ", ".join(msg)

            pbar.set_description(desc=msg)
            pbar.update()
            print()

        pbar.close()

        return True

    # OVERRIDE
    def predict(self, X, device=None, path=None):

        logging.info('')
        logging.info('='*72)
        logging.info("Predict")
        logging.info('='*72)

        # Do not shuffle
        shuffle = False

        # Get/set device
        set_model_device(self, device)

        # Configure training mode
        self.eval()

        # Set number of cores
        torch.set_num_threads(self.num_workers)

        # Create data set
        dataset = self.dataset_class(X, **self.dataset_params, device=device)

        # Create data loader
        dataloader = DataLoader(dataset, shuffle=False, **self.dataloader_params)

        pbar = tqdm(total=int(len(dataloader)/dataloader.batch_size))
        y = []
        for i, (indices, seq_tensor, seq_mask, span_indices, span_mask) in enumerate(dataloader):

            verbose = False

            # Push data through model
            out = self( \
                            seq_tensor = seq_tensor,
                            seq_mask = seq_mask,
                            span_indices = span_indices,
                            span_mask = span_mask,
                            verbose = verbose)

            y_batch = dataset.postprocess_y( \
                                indices = indices,
                                span_scores = out["span_scores"],
                                span_mask = span_mask,
                                role_scores = out["top_role_scores"],
                                role_span_mask = out["top_span_mask"],
                                role_indices = out["top_indices"],
                                )
            y.extend(y_batch)

            pbar.update()
        pbar.close()

        return y

    def loss(self, y_true, y_pred, span_map=None):

        span_loss, role_loss  = self.relation_extractor.loss( \
                            span_labels =       y_true['span_labels'],
                            span_scores =       y_pred['span_scores'],
                            span_mask =         y_true['span_mask'],
                            role_labels =       y_true['role_labels'],
                            top_role_scores =   y_pred['top_role_scores'],
                            top_span_mask =     y_pred['top_span_mask'],
                            top_indices =       y_pred['top_indices'],
                            )

        loss_dict = OrderedDict()
        loss_dict["span_loss"] = span_loss
        loss_dict["role_loss"] = role_loss


        loss = torch.stack([v for k, v in loss_dict.items()])
        loss = aggregate(loss, self.loss_reduction)

        return (loss, loss_dict)

    def perf_counts(self, y_true, y_pred):


        span_counts, role_counts = self.relation_extractor.perf_counts( \
                            span_labels = y_true["span_labels"],
                            span_scores = y_pred["span_scores"],
                            span_mask = y_true["span_mask"],
                            role_labels = y_true["role_labels"],
                            top_role_scores = y_pred["top_role_scores"],
                            top_span_mask = y_pred["top_span_mask"],
                            top_indices = y_pred["top_indices"])

        d = OrderedDict()
        d["span"] = span_counts
        d["role"] = role_counts

        return d
    '''
    def prf(self, y_true, y_pred):


        span_prf, role_prf = self.relation_extractor.prf( \
                            span_labels = y_true["span_labels"],
                            span_scores = y_pred["span_scores"],
                            span_mask = y_true["span_mask"],
                            role_labels = y_true["role_labels"],
                            top_role_scores = y_pred["top_role_scores"],
                            top_span_mask = y_pred["top_span_mask"],
                            top_indices = y_pred["top_indices"])

        prf_dict = OrderedDict()
        prf_dict["span"] = span_prf
        prf_dict["role"] = role_prf

        return prf_dict
    '''
