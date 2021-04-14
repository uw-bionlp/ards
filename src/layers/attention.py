
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from allennlp.modules.attention import BilinearAttention, DotProductAttention
from allennlp.nn.util import weighted_sum
from allennlp.nn.util import masked_softmax
from allennlp.modules import FeedForward

import logging
import numpy as np
import joblib
import os


from layers.activation import get_activation



class Attention(nn.Module):
    '''
    Single-task attention
    '''

    def __init__(self, input_dim, dropout=0.0, use_ffnn=True, query_dim=None,
                                        activation='tanh'):
        super(Attention, self).__init__()

        self.use_ffnn = use_ffnn

        if self.use_ffnn:
            self.ffnn = FeedForward( \
                    input_dim = input_dim,
                    num_layers = 1,
                    hidden_dims = query_dim,
                    activations = get_activation(activation),
                    dropout = 0)
        else:
            query_dim = input_dim

        # Dot product attention
        self.attention = DotProductAttention(normalize=True)

        # Event-specific attention vector
        # (input_dim)
        self.vector = Parameter(torch.Tensor(query_dim))
        torch.nn.init.normal_(self.vector)

        # Dropout
        self.drop_layer = nn.Dropout(p=dropout)


    def forward(self, X, mask=None, verbose=False):
        '''
        Generate predictions


        Parameters
        ----------
        X: input with shape (batch_size, max_seq_len, input_dim)
        mask: input with shape (batch_size, max_seq_len)

        '''



        # Batch size
        batch_size = X.shape[0]

        # Batch vector (repeat across first dimension)
        vector = self.vector.unsqueeze(0).repeat(batch_size, 1)

        #
        if self.use_ffnn:
            Q = self.ffnn(X)
        else:
            Q = X

        # Attention weights
        # shape: (batch_size, max_seq_len)
        alphas = self.attention( \
                                vector = vector,
                                matrix = Q,
                                matrix_mask = mask)

        # Attended input
        # shape: (batch_size, encoder_query_dim)
        output = weighted_sum(X, alphas)

        # Dropout layer
        output = self.drop_layer(output)

        if verbose:
            logging.info('Attention')
            logging.info('\tinput_dim:  {}'.format(input_dim))
            logging.info('\tquery_dim: {}'.format(query_dim))
            logging.info('\tactivation: {}'.format(activation))
            logging.info('\tdropout:    {}'.format(dropout))
            logging.info('\tuse_ffnn:    {}'.format(use_ffnn))

        return (output, alphas)
