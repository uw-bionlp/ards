

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
from models.dataset_xray import DatasetXray, roc
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL
from config.constants import ENTITIES, RELATIONS, DOC_LABELS
from config.constants import PARAMS_FILE, STATE_DICT, PREDICTIONS_FILE
from layers.utils import get_loss, aggregate, PRFAggregator
from layers.plotting import PlotLoss
from scoring.scorer_xray import ScorerXray

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


def nest_dict(D):

    Y = []


    lengths = []
    for k, v in D.items():
        lengths.append(tuple(v.shape)[0])
    assert len(set(lengths)) == 1, "length mismatch: {}".format(lengths)
    length = lengths[0]

    for i in range(length):

        # Loop on dictionary
        d = {k:v[i] for k, v in D.items()}

        # Append to list
        Y.append(d)

    return Y





class ModelXray(Model):

    def __init__(self,  \
        hyperparams,
        dataset_params,
        dataloader_params,
        optimizer_params,
        num_workers,
        num_epochs,
        dataset_class = DatasetXray,
        scorer_class = ScorerXray
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

        # OVERRIDE
        # save model parameters so can be reloaded later
        self.params = OrderedDict()
        self.params['hyperparams'] = hyperparams
        self.params['dataset_params'] = dataset_params
        self.params['dataloader_params'] = dataloader_params
        self.params['optimizer_params'] = optimizer_params
        self.params['num_workers'] = num_workers
        self.params['num_epochs'] = num_epochs




        self.use_rnn = self.hyperparams['use_rnn']
        self.use_doc_classifier = self.hyperparams['use_doc_classifier']
        self.use_span_classifier = self.hyperparams['use_span_classifier']
        self.use_doc_features = self.hyperparams['use_doc_features']


        self.loss_reduction = self.hyperparams['loss_reduction']

        if self.use_rnn:
            self.rnn = RNN(**self.hyperparams['rnn'])

        if self.use_doc_classifier:
            self.doc_classifier = HierarchicalClassifierMulti(**self.hyperparams["doc_classifier"])

        if self.use_span_classifier:
            self.relation_extractor = RelationExtractor(**self.hyperparams['relation_extractor'])

        self.cross_entropy = nn.CrossEntropyLoss()

        self.get_summary()


    #def forward(self, seq_tensor, seq_mask, span_indices, span_mask, span_map=None, verbose=False):
    def forward(self, seq_tensor, seq_mask, verbose=False):
        # seq_tensor (document_count, sentence_count, sentence_length, embed_dim)
        # seq_mask (document_count, sentence_count, sentence_length)
        # span_indices (document_count, span_count, 2)
        # span_mask (document_count, sentence_count, span_count)

        output = OrderedDict()

        '''
        Recurrent layer
        '''
        if self.use_rnn:

            # Iterate over documents - embed with RNN
            H = []
            for st, sm, in zip(seq_tensor, seq_mask):
                # st  (sentence_count, sentence_length, input_dim)
                # sm  (sentence_count, sentence_length)
                # H   (sentence_count, sentence_length, hidden_size*2)
                H.append(self.rnn(st, sm))
        else:
            H = seq_tensor

        '''
        Relation extraction
        '''

        # Iterate over documents - relation extraction
        span_scores = []
        top_role_scores = []
        top_span_mask = []
        top_indices = []
        seq_scores = []
        doc_features = []

        if self.use_span_classifier:
            for i in range(len(H)):
                y = self.relation_extractor( \
                                            seq_tensor = H[i],
                                            span_indices = span_indices[i],
                                            span_mask = span_mask[i],
                                            seq_mask = seq_mask[i],
                                            span_map = span_map,
                                            verbose = verbose,
                                            as_dict = True)

                span_scores.append(y['span_scores'])
                top_role_scores.append(y['top_role_scores'])
                top_span_mask.append(y['top_span_mask'])
                top_indices.append(y['top_indices'])
                doc_features.append(y['doc_vector'])

                seq_scores.append(None)

        output["span_scores"] = span_scores
        output["top_role_scores"] = top_role_scores
        output["top_span_mask"] = top_span_mask
        output["top_indices"] = top_indices
        output["seq_scores"] = seq_scores

        #span_scores = tensor_dict_collect(span_scores)
        #top_role_scores = torch.stack(top_role_scores, dim=0)
        #top_indices = torch.stack(top_indices, dim=0)

        '''
        Document classification
        '''
        doc_scores = []
        sent_scores = []


        if self.use_doc_classifier:

                doc_scores, sent_scores = self.doc_classifier(H, seq_mask)

        output["doc_scores"] = doc_scores
        output["sent_scores"] = sent_scores

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
            #for i, (doc_indices, seq_tensor, seq_mask, span_indices, span_mask, y_true) in enumerate(dataloader):
            for i, (doc_indices, seq_tensor, seq_mask, y_true) in enumerate(dataloader):

                verbose = False #(i == 0) and (j == 0)

                # Reset gradients
                self.zero_grad()

                y_pred = self( \
                                seq_tensor = seq_tensor,
                                seq_mask = seq_mask,
                                #span_indices = span_indices,
                                #span_mask = span_mask,
                                #span_map = dataset.span_mapper.span_map,
                                verbose = verbose)

                loss, loss_dict = self.loss(y_true, y_pred, span_map=dataset.span_mapper.span_map)

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
    def predict(self, X, device=None, path=None, return_prob=False):

        logging.info('')
        logging.info('='*72)
        logging.info("Predict")
        logging.info('='*72)

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
        #for i, (doc_indices, seq_tensor, seq_mask, span_indices, span_mask) in enumerate(dataloader):
        for i, (doc_indices, seq_tensor, seq_mask) in enumerate(dataloader):

            verbose = False

            # Push data through model
            #out = self(seq_tensor, seq_mask, span_indices, span_mask, verbose=verbose)
            out = self(seq_tensor, seq_mask, verbose=verbose)


            if return_prob:
                y_batch = dataset.postprocess_y_prob( \
                                doc_indices = doc_indices,
                                seq_mask  = seq_mask,
                                doc_scores = out["doc_scores"]
                                )
            else:
                y_batch = dataset.postprocess_y( \
                                doc_indices = doc_indices,
                                seq_mask  = seq_mask,
                                doc_scores = out["doc_scores"],
                                sent_scores = out["sent_scores"],
                                #span_scores = out["span_scores"],
                                #span_mask = span_mask,
                                #role_scores = out["top_role_scores"],
                                #role_span_mask = out["top_span_mask"],
                                #role_indices = out["top_indices"],
                                )
            y.extend(y_batch)

            pbar.update()
        pbar.close()


        if path is not None:
            f = os.path.join(path, PREDICTIONS_FILE)
            joblib.dump(y, f)

        return y


    def prob(self, X, y=None, device=None, path=None):

        y_prob = self.predict(X=X, device=device, path=path, return_prob=True)

        if y is not None:
            roc(y, y_prob, path)

        return y_prob

    def loss(self, y_true, y_pred, span_map=None):

        loss_dict = OrderedDict()

        if self.use_doc_classifier:

            doc_loss, sent_loss = self.doc_classifier.loss( \
                            doc_labels = y_true["doc_labels"],
                            doc_scores = y_pred["doc_scores"],
                            sent_labels = y_true["sent_labels"],
                            sent_scores = y_pred["sent_scores"],
                            as_dict = True)

            for k, v in doc_loss.items():
                loss_dict[f"doc_{k}"] = v

            for k, v in sent_loss.items():
                loss_dict[f"sent_{k}"] = v


        if self.use_span_classifier:

            span_labels = nest_dict(y_true["span_labels"])
            span_scores = y_pred["span_scores"]
            span_mask = y_true["span_mask"]
            seq_mask = y_true["seq_mask"]
            seq_scores = y_pred["seq_scores"]
            role_labels = nest_dict(y_true["role_labels"])
            top_role_scores = y_pred["top_role_scores"]
            top_span_mask = y_pred["top_span_mask"]
            top_indices = y_pred["top_indices"]

            span_loss = []
            role_loss = []

            for i in range(len(span_labels)):
                span_ls, role_ls  = self.relation_extractor.loss( \
                                    span_labels = span_labels[i],
                                    span_scores = span_scores[i],
                                    span_mask = span_mask[i],
                                    role_labels = role_labels[i],
                                    top_role_scores = top_role_scores[i],
                                    top_span_mask = top_span_mask[i],
                                    top_indices = top_indices[i],
                                    seq_scores = seq_scores[i],
                                    seq_mask = seq_mask[i],
                                    span_map = span_map
                                    )
                span_loss.append(span_ls)
                role_loss.append(role_ls)

            span_loss = aggregate(torch.stack(span_loss), self.loss_reduction)
            role_loss = aggregate(torch.stack(role_loss), self.loss_reduction)

            loss_dict["span"] = span_loss
            loss_dict["role"] = role_loss


        loss_dict = OrderedDict([(k, v) for k, v in loss_dict.items() if v is not None])
        loss = torch.stack([v for k, v in loss_dict.items()])
        loss = aggregate(loss, self.loss_reduction)
        return (loss, loss_dict)

    def perf_counts(self, y_true, y_pred):


        d = OrderedDict()
        if self.use_doc_classifier:
            doc_counts, sent_counts = self.doc_classifier.perf_counts( \
                            doc_labels = y_true["doc_labels"],
                            doc_scores = y_pred["doc_scores"],
                            sent_labels = y_true["sent_labels"],
                            sent_scores = y_pred["sent_scores"])
            d["doc"] = doc_counts
            d["sent"] = sent_counts


        if self.use_span_classifier:


            span_labels = nest_dict(y_true["span_labels"])
            span_scores = y_pred["span_scores"]
            span_mask = y_true["span_mask"]
            role_labels = nest_dict(y_true["role_labels"])
            top_role_scores = y_pred["top_role_scores"]
            top_span_mask = y_pred["top_span_mask"]
            top_indices = y_pred["top_indices"]

            span_counts = []
            role_counts = []
            for i in range(len(span_labels)):
                sc, rc = self.relation_extractor.perf_counts( \
                                    span_labels = span_labels[i],
                                    span_scores = span_scores[i],
                                    span_mask = span_mask[i],
                                    role_labels = role_labels[i],
                                    top_role_scores = top_role_scores[i],
                                    top_span_mask = top_span_mask[i],
                                    top_indices = top_indices[i]
                                    )
                span_counts.append(sc)
                role_counts.append(rc)

            span_counts = torch.stack(span_counts).sum(dim=0)
            role_counts = torch.stack(role_counts).sum(dim=0)

            d["span"] = span_counts
            d["role"] = role_counts

        return d
