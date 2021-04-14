
import torch
import torch.nn as nn
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules import FeedForward
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

from collections import OrderedDict
import logging


from layers.activation import get_activation
from layers.utils import get_loss, aggregate


from layers.utils import PRF1, PRF1multi, perf_counts_multi, perf_counts


def batched_role_select(labels, indices):
    '''

    Parameters
    ----------
    labels: label ids as tensor (batch_size, span_count_a, span_count_b)
    indices: top indices (batch_size, top span_count)
    '''


    batch_size, span_count_a, span_count_b = tuple(labels.shape)
    if span_count_a != span_count_b:
        NotImplementedError("Currently assumes span counts are the same, need to update to make more general")


    top_labels = []
    for labs, idx in zip(labels, indices):

        # lab: (span_count_a, span_count_b)
        # idx: (top span_count)

        # (top_span_count, span_count)
        labs = torch.index_select(labs, dim=0, index=idx)

        # (top_span_count, top_span_count)
        labs = torch.index_select(labs, dim=1, index=idx)

        top_labels.append(labs)

    # (sequence_count, top_span_count, top_span_count)
    top_labels = torch.stack(top_labels, dim=0)

    return top_labels

def batched_mask_select(mask, indices):
    '''

    Parameters
    ----------
    mask: tensor (batch_size, span_count)
    indices: top indices (batch_size, top span_count)
    '''


    top_labels = []
    for m, idx in zip(mask, indices):

        # m: (span_count)
        # idx: (top span_count)

        m = torch.index_select(m, dim=0, index=idx)
        top_mask.append(m)

    top_mask = torch.cat(top_mask)
    print("top_mask", top_mask.shape)
    a = b

    return top_mask



def get_role_mask(span_mask):

    _, top_count = tuple(span_mask.shape)

    a = span_mask.unsqueeze(1).repeat(1, top_count, 1)
    b = span_mask.unsqueeze(2).repeat(1, 1, top_count)

    role_mask = a * b

    return role_mask





class RoleScorer(nn.Module):
    '''
    Argument scorer learned using feedforward neural networks


    Parameters
    ----------
    label_count: label vocab size

    Returns
    -------
    scores: tensor of scores (batch_size, span_count_a, arg_num, 2)

    '''
    def __init__(self, input_dim, hidden_dim, \
            output_dim = 2,
            activation = 'relu',
            dropout = 0.0,
            num_layers = 1,
            loss_reduction = "sum",
            name = None
            ):


        super(RoleScorer, self).__init__()

        self.output_dim = output_dim
        self.loss_reduction = loss_reduction
        self.name = name


        # Feedforward neural network
        self.ffnn = FeedForward( \
                    input_dim = input_dim,
                    num_layers = num_layers,
                    hidden_dims = hidden_dim,
                    activations = get_activation(activation),
                    dropout = dropout)

        # Score (linear projection)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    #@profile
    def forward(self, a, b, include_embed=False, verbose=False):
        '''
        Parameters
        ----------
        trig_label_scores: tensor of scores (batch_size, span_count_a, trig_tags)
        trig_spans: tensor of spans (batch_size, span_count_a, 2)
        a: tensor of embeddings (batch_size, span_count_a, embed_dim)
        entity_scores: tensor of scores (batch_size, span_count_b, entity_tags)
        entity_spans: tensor of spans (batch_size, span_count_b, 2)
        entity_mask: tensor of mask (batch_size, span_count_b)
        b: tensor of embeddings (batch_size, span_count_b, embed_dim)

        Returns
        -------
        scores: tensor of scores (batch_size, sp, span_count_b, output_dim)
        '''

        # Get size
        batch_size, span_count_a, embed_dim = tuple(a.shape)
        batch_size, span_count_b, embed_dim = tuple(b.shape)

        if span_count_a != span_count_b:
            NotImplementedError("Currently assumes span counts are the same, need to update to make more general")

        '''
        Create trigger-entity embedding pairs
        '''
        # Expand and tile trigger
        # (batch_size, span_count_a, span_count_b, embed_dim)
        a_ = a.unsqueeze(2).repeat(1, 1, span_count_b, 1)

        # Expand and tile entity
        # (batch_size, span_count_a, span_count_b, embed_dim)
        b_ = b.unsqueeze(1).repeat(1, span_count_a, 1, 1)

        # Concatenate trigger and entity embeddings
        # (batch_size, span_count_a, span_count_b, embed_dim*2)
        embedding_pairs = torch.cat((a_, b_), dim=3)

        '''
        Score trigger-entity embedding pairs
        '''
        pair_embed_dim = embedding_pairs.size(-1)

        # Flatten
        # (batch_size*span_count_a*span_count_b, embed_dim*2)
        embedding_pairs = embedding_pairs.view(-1, pair_embed_dim)

        # Push masked embedding_pairs through feedforward neural network
        # (batch_size*span_count_a*span_count_b, hidden_dim)
        projected = self.ffnn(embedding_pairs)

        # argument scores
        # (batch_size*span_count_a*span_count_b, label_count)
        scores = self.linear(projected)

        # Inflate
        # (batch_size, span_count_a, span_count_b, label_count)
        scores = scores.view(batch_size, span_count_a, span_count_b, self.output_dim)

        if verbose:
            logging.info("")
            logging.info(f"RoleScorer")
            logging.info(f"\ta:                    {a.shape}")
            logging.info(f"\tb:                    {b.shape}")
            logging.info(f"\tembeddings, pairwise: {embedding_pairs.shape}")
            logging.info(f"\tscores:               {scores.shape}")


        if include_embed:
            return (scores, embedding_pairs)
        else:
            return scores

    def loss(self, labels, top_scores, top_mask, top_indices):

        # labels (batch_size, span_count_a, span_count_b)
        batch_size, span_count_a, span_count_b = tuple(labels.shape)

        if span_count_a != span_count_b:
            NotImplementedError("Currently assumes span counts are the same, need to update to make more general")

        # scores (batch_size, top span_count_a, top span_count_b, label_count)
        # indices (batch_size, top span_count_a)

        # (batch_size, top_span_count, top_span_count, label_count)
        top_labels = batched_role_select(labels, top_indices)

        # (batch_size, top_span_count, top_span_count)
        role_mask = get_role_mask(top_mask)

        return get_loss(top_labels, top_scores, role_mask, reduction=self.loss_reduction)


    def perf_counts(self, labels, top_scores, top_mask, top_indices):

        # labels (batch_size, span_count_a, span_count_b)
        batch_size, span_count_a, span_count_b = tuple(labels.shape)

        if span_count_a != span_count_b:
            NotImplementedError("Currently assumes span counts are the same, need to update to make more general")

        # scores (batch_size, top span_count_a, top span_count_b, label_count)
        # indices (batch_size, top span_count_a)

        # (batch_size, top_span_count, top_span_count, label_count)
        top_labels = batched_role_select(labels, top_indices)

        # (batch_size, top_span_count, top_span_count)
        role_mask = get_role_mask(top_mask)

        counts = perf_counts(top_labels, top_scores, role_mask)

        return counts

    '''
    def prf(self, labels, top_scores, top_mask, top_indices):

        # labels (batch_size, span_count_a, span_count_b)
        batch_size, span_count_a, span_count_b = tuple(labels.shape)

        if span_count_a != span_count_b:
            NotImplementedError("Currently assumes span counts are the same, need to update to make more general")

        # scores (batch_size, top span_count_a, top span_count_b, label_count)
        # indices (batch_size, top span_count_a)

        # (batch_size, top_span_count, top_span_count, label_count)
        top_labels = batched_role_select(labels, top_indices)

        # (batch_size, top_span_count, top_span_count)
        role_mask = get_role_mask(top_mask)

        prf = PRF1(top_labels, top_scores, role_mask)
        print(prf.tolist(), self.name)

        return prf
    '''


class RoleScorerMulti(nn.Module):
    '''
    Argument scorer learned using feedforward neural networks


    Parameters
    ----------
    label_count: label vocab size

    Returns
    -------
    scores: tensor of scores (batch_size, span_count_a, arg_num, 2)

    '''
    def __init__(self, label_definition, input_dim, hidden_dim, \
            output_dim = 2,
            activation = 'relu',
            dropout = 0.0,
            loss_reduction = "sum"
            ):


        super(RoleScorerMulti, self).__init__()

        self.loss_reduction = loss_reduction

        self.scorers = nn.ModuleDict(OrderedDict())

        for combo in label_definition:
            combo = self.to_key(combo)
            self.scorers[combo] = RoleScorer( \
                        input_dim = input_dim,
                        hidden_dim = hidden_dim,
                        output_dim = output_dim,
                        activation = activation,
                        dropout = dropout,
                        loss_reduction = loss_reduction,
                        name = combo)

    def to_key(self, combo):
        return '-'.join(combo)

    def from_key(self, combo):
        return tuple(combo.split('-'))

    def forward(self, embeddings, include_embed=False, verbose=False):

        scores = OrderedDict()
        embedding_pairs = OrderedDict()
        for k, scorer in self.scorers.items():
            if verbose:
                logging.info("")
                logging.info("SpanScorerMulti: {}".format(k))

            a, b = self.from_key(k)
            scores[(a, b)], embedding_pairs[(a, b)] = scorer( \
                                                a = embeddings[a],
                                                b = embeddings[b],
                                                include_embed = True,
                                                verbose = verbose)

        if include_embed:
            return (scores, embedding_pairs)
        else:
            return scores
            
    def loss(self, labels, top_scores, top_mask, top_indices):

        loss = []
        for k, scorer in self.scorers.items():
            ab = self.from_key(k)
            ls = scorer.loss(labels[ab], top_scores[ab], top_mask, top_indices)
            loss.append(ls)

        loss = aggregate(torch.stack(loss), self.loss_reduction)

        return loss

    def perf_counts(self, labels, top_scores, top_mask, top_indices):


        counts = []
        for k, scorer in self.scorers.items():
            ab = self.from_key(k)

            counts_tmp = scorer.perf_counts(labels[ab], top_scores[ab], top_mask, top_indices)

            counts.append(counts_tmp)

        # precision,recall,and f1 as tensor of size (3)
        counts = torch.stack(counts).sum(dim=0)

        return counts


    '''
    def prf(self, labels, top_scores, top_mask, top_indices):


        prf = []
        for k, scorer in self.scorers.items():
            ab = self.from_key(k)

            # precision,recall,and f1 as tensor of size (3)
            prf_tmp = scorer.prf(labels[ab], top_scores[ab], top_mask, top_indices)
            #print(k, prf_tmp.tolist())
            prf.append(prf_tmp)

        # precision,recall,and f1 as tensor of size (3)
        prf = torch.stack(prf).mean(dim=0)

        return prf
    '''
