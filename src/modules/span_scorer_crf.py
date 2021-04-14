

import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from multiprocessing import Pool
import numpy as np
import time

from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules import FeedForward
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules import TimeDistributed
from allennlp.nn import util




from layers.activation import get_activation
from layers.utils import get_loss, aggregate
from layers.utils import PRF1, PRF1multi


B_PREFIX = 'B'
I_PREFIX = 'I'
O_PREFIX = 'O'


def label_map(num_tags):


    n = num_tags
    n_pos = n - 1

    span_to_seq = OrderedDict()
    seq_to_span = OrderedDict()

    for i in range(num_tags):

        if i == 0:
            O = 0
            span_to_seq[i] = (O, O)
        else:
            B = i
            I = i + n_pos
            span_to_seq[i] = (B, I)


        if i == 0:
            O = 0
            seq_to_span[O] = (O_PREFIX, O)
        else:
            B = i
            I = i + n_pos

            assert B not in seq_to_span
            seq_to_span[B] = (B_PREFIX, B)

            assert I not in seq_to_span
            seq_to_span[I] = (I_PREFIX, B)


    logging.info('-'*72)
    logging.info('Sequence tag-ID mapping')
    logging.info('-'*72)

    logging.info('')
    logging.info('Span to Sequence map:')
    for k, v in span_to_seq.items():
        logging.info('{} --> {}'.format(k, v))

    logging.info('')
    logging.info('Sequence to Span map:')
    for k, v in seq_to_span.items():
        logging.info('{} --> {}'.format(k, v))

    return (span_to_seq, seq_to_span)






def tag_to_span_lab(seq_label, num_tags_orig):
    '''
    Convert BIO representation to span representation


    Parameters
    ----------
    seq_label: int, span label with BIO indices
    num_tags: int, number of original tags, without BI prefixes
    '''

    is_O = 0
    is_B = 0
    is_I = 0

    # Number of positive (non-negative tags)
    num_pos_tags = num_tags_orig - 1

    # Convert sequence label to span label (i.e. convert BI)
    if seq_label > num_pos_tags:
        span_label = seq_label - num_pos_tags
        is_I = 1
    elif seq_label > 0:
        span_label = seq_label
        is_B = 1
    else:
        span_label = seq_label
        is_O = 1

    return (span_label, is_O, is_B, is_I)



#def BIO_to_span(labels, id_to_label=None, lab_is_tuple=True,
#                     num_tags_orig=None, tag_to_lab_fn=None):
def BIO_to_span(labels, seq_tag_map):
    '''

    Finds spans in BIO sequence

    NOTE: start span index is inclusive, end span index is exclusive
            e.g. like Python lists

    Parameters
    ----------
    labels: list of token label ids (tag ids)
    '''

    spans = []
    begin_count = 0
    start = -1
    end = -1
    active_tag = None

    # No non-negative labels, so return empty list
    if not any(labels):
        return []

    # Loop on tokens in seq
    for i, lab in enumerate(labels):

        #if tag_to_lab_fn is not None:
        #    tag, is_O, is_B, is_I = tag_to_lab_fn(lab)

        # Label is tuple
        #elif lab_is_tuple:
        #    # Convert current sequence tag label to span label
        #    if id_to_label is None:
        #        prefix, tag  = lab
        #    else:
        #        prefix, tag  = id_to_label[lab]

        #    is_O = prefix == OUTSIDE
        #    is_B = prefix == BEGIN
        #    is_I = prefix == INSIDE
        prefix, tag = seq_tag_map[lab]

        is_O = prefix == O_PREFIX
        is_B = prefix == B_PREFIX
        is_I = prefix == I_PREFIX


        # Label is not tuple, so use number of tags to resolve BI
        # prefixes
        #else:
        #    assert num_tags_orig is not None
        #    tag, is_O, is_B, is_I = tag_to_span_lab(lab, num_tags_orig)



        # Outside label
        if is_O:

            # The span has ended
            if active_tag is not None:
                spans.append((active_tag, start, end))

            # Not in a span
            active_tag = None

        # Span beginning
        elif is_B:

            # The span has ended
            if active_tag is not None:
                spans.append((active_tag, start, end))

            # Update active tag
            active_tag = tag

            # Index of current span start
            start = i
            end = i + 1

            # Increment begin count
            begin_count += 1

        # Span inside and current tag matches active tag
        # e.g. well-formed span
        elif is_I and (tag == active_tag):
            end += 1

        # Ill formed span
        elif is_I and (tag != active_tag):

            # Capture end of valid span
            if active_tag is not None:
                spans.append((active_tag, start, end))

            # Not in a span
            active_tag = None

        else:
            raise ValueError("could not assign label")

    # Last token might be part of a valid span
    if active_tag is not None:
        spans.append((active_tag, start, end))

    # Get span count
    span_count = len(spans)

    if True and (begin_count != span_count):
        msg = \
        '''Count mismatch:
        seq = {}
        Begin count = {}
        span count = {}'''.format(seq, begin_count, span_count)
        logging.warn(msg)

    return spans

def seq_tags_to_spans(seq_tags, span_map, seq_tag_map):
    '''
    Convert sequence tags to span labels

    Parameters
    ----------
    seq_tags: list of list of label indices
             i.e. list of sentences, where each sentence
                  is a list of label indices

    Returns
    -------
    span_labels: tensor of shape (batch_size, num_spans)

    '''

    #start_time = time.time()

    # Get inputs for tensor initialization
    batch_size = len(seq_tags)
    num_spans = len(span_map)

    # Initialize span labels to null label
    span_labels = torch.zeros(batch_size, num_spans).type( \
                                                   torch.LongTensor)

    # Loop on sequences
    for i_seq, seq in enumerate(seq_tags):

        # Convert BIO to spans
        S = BIO_to_span(seq, seq_tag_map)

        # Iterate over spans
        for lab, start, end in S:

            # Token indices of current span
            idx = (start, end)

            # Span in map
            if idx in span_map:

                # Span index within tensor
                i_span = span_map[idx]

                # Update label tensor
                span_labels[i_seq, i_span] = lab

            # Span not in map
            else:
                logging.warn("span not in map:\t{}".format(idx))

    return span_labels



def get_seq_labels(span_labels, span_map, span_to_seq, max_len):

    '''
    Get tensor representation for mention (e.g. trigger)

    Parameters
    ----------



    '''



    # Create reverse span map, mapping span indices to token indices
    rev_span_map = OrderedDict([(i, s) for s, i in span_map.items()])


    # Sentence count
    batch_size, span_count = tuple(span_labels.shape)


    #span_labels = span_labels.cpu()
    #span_labels = span_labels.tolist()

    # Loop on sentences in document
    seq_labels = torch.zeros((batch_size, max_len)).long()
    seq_labels = seq_labels.to(span_labels.get_device())
    span_labels = span_labels.cpu().numpy()


    # Iterate over sequences
    for i, spans in enumerate(span_labels):

        # Iterate over spans in sequence
        for j, span_lab in enumerate(spans):

            # Gets token indices
            start, end = rev_span_map[j]

            # Get applicable begin/inside labels
            B, I = span_to_seq[span_lab]

            # Update sequence labels
            seq_labels[i, start:end] = I
            seq_labels[i, start] = B

    return seq_labels


class SpanScorerCRF(nn.Module):
    '''
    Span extractor
    '''
    def __init__(self, input_dim, num_tags,
            low_val = -5,
            high_val = 5,
            incl_start_end = True,
            name = None,
            ):
        super(SpanScorerCRF, self).__init__()

        self.input_dim = input_dim
        self.num_tags = num_tags
        self.low_val = low_val
        self.high_val = high_val
        self.incl_start_end = incl_start_end
        self.name = name


        self.span_to_seq, self.seq_to_span = label_map(num_tags)

        self.num_tags_seq = len(self.seq_to_span)
        self.num_tags_span = len(self.span_to_seq)

        # Linear projection layer
        self.projection = nn.Linear(input_dim, self.num_tags_seq)

        # Create event-specific CRF
        self.crf = ConditionalRandomField( \
                        num_tags = self.num_tags_seq,
                        include_start_end_transitions = incl_start_end)

    def forward(self, seq_tensor, seq_mask, span_map, span_indices, verbose=False):
        '''
        Calculate logits
        '''
        # Dimensionality
        batch_size, max_seq_len, input_dim = tuple(seq_tensor.shape)

        # Project input tensor sequence to logits
        seq_scores = self.projection(seq_tensor)

        '''
        Decoding sequence tags
        '''

        # Viterbi decode
        best_paths = self.crf.viterbi_tags( \
                                        logits = seq_scores,
                                        mask = seq_mask)
        seq_pred, score = zip(*best_paths)
        seq_pred = list(seq_pred)

        '''
        Convert sequence tags to span predictions
        '''
        # Get spans from sequence tags
        #   Converts list of list of predicted label indices to
        #   tensor of size (batch_size, num_spans)
        span_pred = seq_tags_to_spans( \
                                seq_tags = seq_pred,
                                span_map = span_map,
                                seq_tag_map = self.seq_to_span)


        span_pred = span_pred.to(seq_tensor.device)

        # Get scores from labels
        span_pred = F.one_hot(span_pred, num_classes=self.num_tags_span).float()

        #print('crf seq  pos: ', sum([int(w > 0) for W in seq_pred for w in W]))
        #print('crf span pos: ', (span_pred > 0).sum().tolist())
        #print(span_pred)
        return (seq_scores, span_pred)

    def loss(self, span_labels, seq_scores, seq_mask, span_map):



        batch_size, max_len, embed_dim = tuple(seq_scores.shape)



        seq_labels = get_seq_labels( \
                        span_labels = span_labels,
                        span_map = span_map,
                        span_to_seq = self.span_to_seq,
                        max_len = max_len)

        seq_mask[:,0] = True


        # Get loss (negative log likelihood)
        loss = -self.crf( \
                            inputs = seq_scores,
                            tags = seq_labels,
                            mask = seq_mask)
        #print('loss', loss)

        return loss



class SpanScorerCRFMulti(nn.Module):
    '''
    Span scorer


    Parameters
    ----------
    num_tags: label vocab size


    Returns
    -------
    arg_scores: tensor of scores (batch_size, trig_num, arg_num)

    '''
    def __init__(self, label_definition, input_dim, loss_reduction='sum'):


        super(SpanScorerCRFMulti, self).__init__()

        self.loss_reduction = loss_reduction
        self.scorers = nn.ModuleDict(OrderedDict())

        for k, label_set in label_definition.items():
            self.scorers[k] = SpanScorerCRF( \
                        input_dim = input_dim,
                        num_tags = len(label_set),
                        name = k)

        self.types = self.scorers.keys()

    def forward(self, seq_tensor, seq_mask, span_map, span_indices, verbose=False):

        seq_scores = OrderedDict()
        span_pred = OrderedDict()
        for k, scorer in self.scorers.items():
            if verbose:
                logging.info("")
                logging.info("SpanScorerCRFMulti: {}".format(k))


            seq_scores[k], span_pred[k] = scorer( \
                                    seq_tensor = seq_tensor,
                                    seq_mask = seq_mask,
                                    span_map = span_map,
                                    span_indices = span_indices,
                                    verbose = verbose)

        return (seq_scores, span_pred)



    def loss(self, span_labels, seq_scores, seq_mask, span_map):

        loss = []
        for k, scorer in self.scorers.items():
            ls = scorer.loss(span_labels[k], seq_scores[k], seq_mask, span_map)
            loss.append(ls)

        loss = aggregate(torch.stack(loss), self.loss_reduction)

        return loss

    def prf(self, labels, scores, mask):

        # precision,recall,and f1 as tensor of size (3)
        prf = PRF1multi(labels, scores)
        return prf
