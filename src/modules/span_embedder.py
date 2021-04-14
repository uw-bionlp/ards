import torch
import torch.nn as nn
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules import FeedForward
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

from collections import OrderedDict
import logging

from layers.activation import get_activation

class SpanEmbedder(nn.Module):
    '''
    Create span embeddings


    Parameters
    ----------
    num_tags: label vocab size

    Returns
    -------
    arg_scores: tensor of scores (batch_size, trig_num, arg_num, 2)

    '''
    def __init__(self, input_dim, \

            # FFNN projection parameters
            project = True,
            hidden_dim = 100,
            activation = 'tanh',
            dropout = 0.0,

            # General config
            span_end_is_exclusive = True,

            ):

        super(SpanEmbedder, self).__init__()

        # Span end is exclusive (like Python or C)
        self.span_end_is_exclusive = bool(span_end_is_exclusive)

        '''
        Self-attentive span extractor
        '''
        # Create extractor
        self.extractor = SelfAttentiveSpanExtractor( \
                            input_dim = input_dim)

        '''
        Nonlinear projection via feedforward neural network
        '''
        self.project = project
        if self.project:
            self.ffnn = FeedForward( \
                    input_dim = input_dim,
                    num_layers = 1,
                    hidden_dims = hidden_dim,
                    activations = get_activation(activation),
                    dropout = dropout)
            self.output_dim = hidden_dim
        else:
            self.output_dim = input_dim



    def forward(self, seq_tensor, span_indices, span_mask, verbose=False):
        '''
        Parameters
        ----------
        seq_tensor: sequence representation (batch_size, seq_len, embed_dim)
        seq_mask: sequence mask (batch_size, seq_len)
        span_indices: tensor of span indices (batch_size, span_num, 2)
        span_mask: tensor of mask (batch_size, trig_num)

        Returns
        -------
        embeddings: tensor of span embeddings (batch_size, span_num, output_dim)
        '''



        # If span end indices are exclusive, subtract 1
        if self.span_end_is_exclusive:
            starts, ends = span_indices.split(1, dim=-1)
            span_indices = torch.cat((starts, ends-1), dim=-1)


        embedding = self.extractor( \
                                sequence_tensor = seq_tensor,
                                span_indices = span_indices,
                                span_indices_mask = span_mask)


        # Project embedding
        if self.project:
            projection = self.ffnn(embedding)
        else:
            projection = embedding





        if verbose:
            logging.info("")
            logging.info('SpanEmbedder')
            logging.info('\tembedding:  {}'.format(embedding.shape))
            logging.info('\tprojection: {}'.format(projection.shape))

        return projection


class SpanEmbedderMulti(nn.Module):

    def __init__(self, label_definition, input_dim, \
            project = False,
            hidden_dim = None,
            activation = 'tanh',
            dropout = 0.0,
            span_end_is_exclusive = True,
            ):

        super(SpanEmbedderMulti, self).__init__()

        self.embedders = nn.ModuleDict(OrderedDict())
        for k, lab_def in label_definition.items():
            self.embedders[k] = SpanEmbedder( \
                        input_dim = input_dim,
                        project = project,
                        hidden_dim = hidden_dim,
                        activation = activation,
                        dropout = dropout,
                        span_end_is_exclusive = True)


        self.output_dim = self.embedders[k].output_dim

    def forward(self, seq_tensor, span_indices, span_mask, verbose=False):

        embeddings = OrderedDict()
        for k, embedder in self.embedders.items():
            if verbose:
                logging.info("SpanEmbedderMulti: {}".format(k))
            embeddings[k] = embedder( \
                            seq_tensor = seq_tensor,
                            span_indices = span_indices,
                            span_mask = span_mask,
                            verbose = verbose)
        return embeddings
