import torch
import torch.nn as nn
from allennlp.modules import FeedForward
from allennlp.modules.time_distributed import TimeDistributed
from collections import OrderedDict
from allennlp.nn import util
import torch.nn.functional as F


import logging

from layers.activation import get_activation
from layers.utils import get_loss, aggregate


from layers.utils import PRF1, PRF1multi, perf_counts_multi


class SpanScorer(nn.Module):
    '''
    Span scorer


    Parameters
    ----------
    num_tags: label vocab size


    Returns
    -------
    arg_scores: tensor of scores (batch_size, trig_num, arg_num, num_tags)

    '''
    def __init__(self, input_dim, hidden_dim, num_tags, \
            activation = 'relu',
            dropout = 0.0,
            loss_reduction = 'sum',
            name = None,
            class_weights = None):
        super(SpanScorer, self).__init__()


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.name = name


        self.activation = activation
        self.activation_fn = get_activation(activation)
        self.dropout = dropout
        self.loss_reduction = loss_reduction
        self.num_layers = 1

        '''
        Create classifier
        '''

        # Feedforward neural network for predicting span labels
        self.FF = FeedForward( \
                    input_dim = self.input_dim,
                    num_layers = self.num_layers,
                    hidden_dims = self.hidden_dim,
                    activations = self.activation_fn,
                    dropout = self.dropout)

        # Span classifier
        self.scorer = torch.nn.Sequential(
            TimeDistributed(self.FF),
            TimeDistributed(torch.nn.Linear(self.hidden_dim, self.num_tags)))

        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = torch.tensor(class_weights, requires_grad=False)
        #self.register_buffer('class_weights', self.class_weights)



    def forward(self, embedding, verbose=False):

        '''
        Parameters
        ----------
        embed: (batch_size, num_spans, input_dim)
        mask: span mask (batch_size, num_spans)
        labels: true span labels (batch_size, num_spans)

        seq_feat: sequence-level features (batch_size, seq_feat_size)

        Returns
        -------
        '''


        # Compute label scores
        # (batch_size, num_spans, num_tags)
        scores = self.scorer(embedding)

        if verbose:
            logging.info("")
            logging.info('SpanScorer')
            logging.info('\tembedding: {}'.format(embedding.shape))
            logging.info('\tscores:    {}'.format(scores.shape))

        return scores

    def loss(self, labels, scores, mask):

        '''

        pred = scores.max(-1)[1]

        tag_num = scores.size(-1)
        scores_flat = scores.view(-1, tag_num)
        # (batch_size*trig_num*entity_num)
        labels_flat = labels.view(-1)
        # (batch_size*trig_num*entity_num)
        mask_flat = mask.view(-1).bool()

        scores_mask = scores_flat[mask_flat]
        labels_mask = labels_flat[mask_flat]


        if self.name == 'region':


            print('a', labels[0][24], scores[0][24], mask[0][24])
            print('b', labels[0][121], scores[0][121], mask[0][121])
            print('c', labels[1][9], scores[1][9], mask[1][9])


            for i, (s, l) in enumerate(zip(scores_mask, labels_mask)):
                #b_ = b_.tolist()
                #a_ = a_.tolist()

                if l:
                    print('flat', i, l, s, F.cross_entropy(s.unsqueeze(0), l.unsqueeze(0), reduction='sum').tolist(), F.cross_entropy(s.unsqueeze(0), l.unsqueeze(0), reduction='sum', weight=self.class_weights).tolist())

            print('cross', F.cross_entropy(scores_mask, labels_mask, reduction='sum').tolist())
            print('cross', F.cross_entropy(scores_mask, labels_mask, reduction='sum', weight=self.class_weights).tolist())

            print('-'*10)
            print(self.name)
            assert (labels > 0).sum().tolist() == (labels*mask > 0).sum().tolist()
            print('mask pos', mask.sum().tolist())
            print('mask flat', mask_flat.shape)
            print('scores flat', scores_mask.shape)
            print('labels flat', labels_mask.shape)
            print('labels pos1', (labels > 0).sum().tolist())
            print('labels pos2', (labels_flat > 0).sum().tolist())
            print('scores pos ',  (pred > 0).sum().tolist())
            #for s, l in zip(scores, labels):
            #    print(l)
            #    print(s)
        '''

        if (self.class_weights is not None) and (self.class_weights.get_device() != labels.get_device()):
            self.class_weights = self.class_weights.to(labels.get_device())

        loss = get_loss( \
                labels = labels,
                scores = scores,
                mask = mask,
                reduction = self.loss_reduction,
                weight = self.class_weights)
        #if self.name == 'region':
        #    print('loss', loss)
        return loss

class SpanScorerMulti(nn.Module):
    '''
    Span scorer


    Parameters
    ----------
    num_tags: label vocab size


    Returns
    -------
    arg_scores: tensor of scores (batch_size, trig_num, arg_num)

    '''
    def __init__(self, label_definition, input_dim, hidden_dim, \
            activation = 'relu',
            dropout = 0.0,
            loss_reduction = 'sum',
            class_weights = None):
        super(SpanScorerMulti, self).__init__()

        self.loss_reduction = loss_reduction

        self.scorers = nn.ModuleDict(OrderedDict())

        for k, label_set in label_definition.items():

            if class_weights is None:
                cw = None
            else:
                cw = class_weights[k]

            self.scorers[k] = SpanScorer( \
                        input_dim = input_dim,
                        hidden_dim = hidden_dim,
                        num_tags = len(label_set),
                        activation = activation,
                        dropout = dropout,
                        loss_reduction = loss_reduction,
                        class_weights = cw,
                        name = k)

        self.types = self.scorers.keys()

    def forward(self, embeddings, mask, verbose=False):

        scores = OrderedDict()
        for k, scorer in self.scorers.items():
            if verbose:
                logging.info("")
                logging.info("SpanScorerMulti: {}".format(k))

            # input is dict - get scorer hyphen specific embeddings
            if isinstance(embeddings, dict):
                embed = embeddings[k]

            # input is tensor - use same embeddings for all scorers
            else:
                embed = embeddings

            scores[k] = scorer( \
                        embedding = embed,
                        verbose = verbose)


        return scores



    def loss(self, labels, scores, mask):

        loss = []
        for k, scorer in self.scorers.items():
            ls = scorer.loss(labels[k], scores[k], mask)
            loss.append(ls)


            _, pred = scores[k].max(-1)
            true = (labels[k] > 0).sum().tolist()
            pos = (pred > 0).sum().tolist()


        loss = aggregate(torch.stack(loss), self.loss_reduction)

        return loss

    def perf_counts(self, labels, scores, mask):
        return perf_counts_multi(labels, scores, mask)

    '''
    def prf(self, labels, scores, mask):

        # precision,recall,and f1 as tensor of size (3)
        prf = PRF1multi(labels, scores, mask)
        return prf
    '''
