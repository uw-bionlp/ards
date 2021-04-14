

from collections import Counter, OrderedDict
import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from allennlp.modules import FeedForward
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_

from allennlp.modules import FeedForward
from allennlp.nn.activations import Activation

import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import joblib
import math
import os


from layers.attention import Attention
from layers.activation import get_activation
from layers.utils import get_loss, aggregate
from layers.utils import PRF1, PRF1multi, perf_counts_multi, perf_counts, default_counts_tensor
from utils.misc import nest_list, nest_dict

class HierarchicalClassifier(nn.Module):
    '''

    '''
    def __init__(self, \
            input_dim,
            query_dim = None,
            projection_dim = 100,
            doc_num_tags = None,
            use_ffnn = True,
            dropout_sent_classifier = 0.0,
            dropout_doc_classifier = 0.0,
            activation = 'tanh',
            loss_reduction = 'sum',
            use_sent_objective = False,
            concat_sent_scores = False,
            sent_definition = None,

            #use_ffnn = True,
            #ffnn_hidden_dim = 50,
            #ffnn_dropout = 0.0,
            #ffnn_activation_fn = 'tanh'
            ):

        super(HierarchicalClassifier, self).__init__()


        self.use_sent_objective = use_sent_objective
        self.concat_sent_scores = concat_sent_scores
        self.sent_definition = sent_definition
        #self.use_ffnn = use_ffnn


        if use_ffnn:
            assert query_dim is not None
            assert doc_num_tags is not None
        else:
            if query_dim is not None:
                logging.warn(f"Overriding query_dim to: {input_dim}")
            query_dim = input_dim

        self.loss_reduction = loss_reduction


        self.word_attention = Attention( \
                                input_dim = input_dim,
                                dropout = 0,
                                use_ffnn = use_ffnn,
                                query_dim = query_dim,
                                activation = activation,
                                )


        if self.use_sent_objective:
            self.sent_classifiers = SentClassifiers( \
                                input_dim = input_dim,
                                num_tags = 2,
                                loss_reduction = loss_reduction,
                                dropout = dropout_sent_classifier,
                                include_projection = True,
                                projection_dim = projection_dim,
                                sent_definition = sent_definition)

            input_dim += self.sent_classifiers.output_dim*int(self.concat_sent_scores)


        self.doc_ffnn = FeedForward( \
                input_dim = input_dim,
                num_layers = 1,
                hidden_dims = projection_dim,
                activations = get_activation('tanh'),
                dropout = dropout_doc_classifier)

        self.sent_attention = Attention( \
                                input_dim = projection_dim,
                                dropout = 0,
                                use_ffnn = use_ffnn,
                                query_dim = query_dim,
                                activation = activation,
                                )

        self.output_layer = nn.Linear(projection_dim, doc_num_tags)


    def forward(self, seq_tensor, seq_mask, verbose=False):

        # seq_tensor (document_count, sentence_count, sentence_length, embed_dim)
        # seq_mask (document_count, sentence_count, sentence_length)

        # Iterate over documents
        sent_vectors = []
        sent_masks = []
        sent_scores = []
        for seq, mask in zip(seq_tensor, seq_mask):

            # seq_tensor_ (sentence_count, sentence_length, embed_dim)
            # seq_mask_ (sentence_count, sentence_length)

            # v (sentence_count, hidden_size)
            # alphas (sentence_count, sentence_length)
            sent_vec, alphas = self.word_attention(seq, mask)


            if self.use_sent_objective:
                # dict of scores
                sent_scores_dict = self.sent_classifiers(sent_vec)
                sent_scores.append(sent_scores_dict)

                if self.concat_sent_scores:
                    # (sentence_count, sent_classifiers.output_dim)
                    sent_scores_flat = [v for _, v in sent_scores_dict.items()]
                    sent_scores_flat.append(sent_vec)
                    sent_vec = torch.cat(sent_scores_flat, dim=1)

            sent_vectors.append(sent_vec)

            # sent_mask (sentence_count)
            sent_msk = torch.sum(mask, dim=1) > 0
            sent_masks.append(sent_msk)


        if self.use_sent_objective:
            sent_scores = nest_list(sent_scores)
            for k, v in sent_scores.items():
                # (document_count, sentence_count, 2)
                sent_scores[k] = torch.stack(v)
        else:
            sent_scores = None



        # sent_vectors (document_count, sentence_count, hidden_size)
        sent_vectors = torch.stack(sent_vectors)

        # sent_vectors (document_count, sentence_count, projection_dim)
        sent_vectors = self.doc_ffnn(sent_vectors)

        # sent_masks (document_count, sentence_count)
        sent_masks = torch.stack(sent_masks)

        # alphas (document_count, sentence_count)
        # doc_vectors (document_count, hidden_size)
        doc_vectors, alphas = self.sent_attention(sent_vectors, sent_masks)

        #if self.use_ffnn:
        #    doc_scores = self.FFNN(doc_vectors)

        doc_scores = self.output_layer(doc_vectors)

        return (doc_scores, sent_scores)

    def loss(self, doc_labels, doc_scores, sent_labels=None, sent_scores=None):

        doc_loss = F.cross_entropy( \
                            input = doc_scores,
                            target = doc_labels,
                            reduction = self.loss_reduction)

        if self.use_sent_objective:
            sent_loss = self.sent_classifiers.loss( \
                                labels = sent_labels,
                                scores = sent_scores)
        else:
            sent_loss = None


        return (doc_loss, sent_loss)

    def prf(self, doc_labels, doc_scores, sent_labels=None, sent_scores=None):
        return PRF1(doc_labels, doc_scores)

    def perf_counts(self, doc_labels, doc_scores, sent_labels=None, sent_scores=None):

        doc_counts = perf_counts(doc_labels, doc_scores)

        if self.use_sent_objective:
            sent_counts = self.sent_classifiers.perf_counts(sent_labels, sent_scores)
        else:
            sent_counts = default_counts_tensor()

        return (doc_counts, sent_counts)



class HierarchicalClassifierMulti(nn.Module):
    '''



    Parameters
    ----------


    Returns
    -------


    '''
    def __init__(self,
            doc_definition,
            input_dim,
            query_dim = None,
            use_ffnn = True,
            dropout_sent_classifier = 0.0,
            dropout_doc_classifier = 0.0,
            activation = 'tanh',
            loss_reduction = "sum",
            use_sent_objective = False,
            concat_sent_scores = False,
            sent_definition = None,
            projection_dim = 100,
            #use_ffnn = True,
            #ffnn_hidden_dim = 50,
            #ffnn_dropout = 0.0,
            #ffnn_activation_fn = 'tanh'
            ):


        super(HierarchicalClassifierMulti, self).__init__()

        self.loss_reduction = loss_reduction
        self.use_sent_objective = use_sent_objective

        self.classifiers = nn.ModuleDict(OrderedDict())

        for k, label_set in doc_definition.items():

            self.classifiers[k] = HierarchicalClassifier( \
                        input_dim = input_dim,
                        query_dim = query_dim,
                        doc_num_tags = len(label_set),
                        use_ffnn = use_ffnn,
                        dropout_sent_classifier = dropout_sent_classifier,
                        dropout_doc_classifier = dropout_doc_classifier,
                        activation = activation,
                        loss_reduction = loss_reduction,
                        use_sent_objective = use_sent_objective,
                        concat_sent_scores = concat_sent_scores,
                        sent_definition = sent_definition[k],
                        projection_dim = projection_dim,
                        )

    def forward(self, seq_tensor, seq_mask, verbose=False):

        doc_scores = OrderedDict()
        sent_scores = OrderedDict()
        for k, classifier in self.classifiers.items():
            doc_scores[k], sent_scores[k] = classifier( \
                                                    seq_tensor = seq_tensor,
                                                    seq_mask = seq_mask,
                                                    verbose = verbose)

        if not self.use_sent_objective:
            sent_scores = None

        return (doc_scores, sent_scores)

    def loss(self, doc_labels, doc_scores, sent_labels=None, sent_scores=None, as_dict=False):



        doc_loss = OrderedDict()
        sent_loss = OrderedDict()
        for k, classifier in self.classifiers.items():

            dl, sl = classifier.loss( \
                    doc_labels = doc_labels[k],
                    doc_scores = doc_scores[k],
                    sent_labels = None if sent_labels is None else sent_labels[k],
                    sent_scores = None if sent_scores is None else sent_scores[k])

            doc_loss[k] = dl
            sent_loss[k] = sl


        if as_dict:
            return (doc_loss, sent_loss)

        else:
            doc_loss = [v for k, v in doc_loss.items()]
            doc_loss = aggregate(torch.stack(doc_loss), self.loss_reduction)

            if self.use_sent_objective:
                sent_loss = [v for k, v in sent_loss.items()]
                sent_loss = aggregate(torch.stack(sent_loss), self.loss_reduction)

            else:
                sent_loss = None

        return (doc_loss, sent_loss)

    #def prf(self, doc_labels, doc_scores, sent_labels=None, sent_scores=None):
    #    prf = PRF1multi(doc_labels, doc_scores)
    #    return prf

    def perf_counts(self, doc_labels, doc_scores, sent_labels=None, sent_scores=None):

        doc_counts = default_counts_tensor()
        sent_counts = default_counts_tensor()
        for k, classifier in self.classifiers.items():
            dc, sc = classifier.perf_counts( \
                    doc_labels = doc_labels[k],
                    doc_scores = doc_scores[k],
                    sent_labels = None if sent_labels is None else sent_labels[k],
                    sent_scores = None if sent_scores is None else sent_scores[k])
            doc_counts += dc
            sent_counts += sc

        return (doc_counts, sent_counts)



class SentClassifier(nn.Module):
    '''

    '''
    def __init__(self, \
            input_dim,
            num_tags = 2,
            loss_reduction = 'sum',
            dropout = 0.0,
            include_projection = False,
            projection_dim = 100
            ):

        super(SentClassifier, self).__init__()

        self.num_tags = num_tags
        self.loss_reduction = loss_reduction
        self.include_projection = include_projection

        if self.include_projection:
            self.ffnn = FeedForward( \
                    input_dim = input_dim,
                    num_layers = 1,
                    hidden_dims = projection_dim,
                    activations = get_activation('tanh'),
                    dropout = 0)
            linear_input_dim = projection_dim
        else:
            linear_input_dim = input_dim

        self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(linear_input_dim, num_tags)


    def forward(self, X):

        # nonlinear projection
        if self.include_projection:
            X = self.ffnn(X)

        # apply dropout
        X = self.dropout(X)

        # output linear layer
        out = self.linear(X)

        return out

    def loss(self, labels, scores):

        # (document count*sentence count) from  (document count, sentence count)
        labels = labels.view(-1)

        # (document count*sentence count, num_tags) from  (document count, sentence count, num_tags)
        scores = scores.view(-1, self.num_tags)

        loss = F.cross_entropy(scores, labels, reduction=self.loss_reduction)

        return loss


    def perf_counts(self, labels, scores):
        NotImplementedError("need to add perf_counts method")

class SentClassifiers(nn.Module):
    '''

    '''
    def __init__(self, \
            input_dim,
            num_tags = 2,
            loss_reduction = 'sum',
            dropout = 0.0,
            include_projection = False,
            projection_dim = 100,
            sent_definition = None,
            ):

        super(SentClassifiers, self).__init__()

        self.loss_reduction = loss_reduction
        self.output_dim = len(sent_definition)*num_tags


        self.classifiers = nn.ModuleDict(OrderedDict())
        for k in sent_definition:
            k = self.to_key(k)
            self.classifiers[k] = SentClassifier( \
                                    input_dim = input_dim,
                                    num_tags = num_tags,
                                    loss_reduction = loss_reduction,
                                    dropout = dropout,
                                    include_projection = include_projection,
                                    projection_dim = projection_dim)


    def to_key(self, combo):
        return '-'.join(combo)

    def from_key(self, combo):
        return tuple(combo.split('-'))

    def forward(self, X):

        scores = OrderedDict()
        for k, classifier in self.classifiers.items():
            k = self.from_key(k)
            scores[k] = classifier(X)


        return scores

    def loss(self, labels, scores):

        loss = []
        for k, classifier in self.classifiers.items():
            k = self.from_key(k)
            ls = classifier.loss(labels[k], scores[k])
            loss.append(ls)

        loss = aggregate(torch.stack(loss), self.loss_reduction)

        return loss

    def perf_counts(self, labels, scores):

        counts = default_counts_tensor()
        for k, classifier in self.classifiers.items():
            k = self.from_key(k)
            counts += perf_counts(labels[k], scores[k])

        return counts
