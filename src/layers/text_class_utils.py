

import torch
import torch.nn.functional as F
from collections import OrderedDict, Counter
import logging



#from layers.xfmr import char2wordpiece, wordpiece2char
from layers.xfmr2 import char2token_idx, token2char_idx

from layers.padding import  pad3D, pad2D
from layers.utils import set_model_device, set_tensor_device
from layers.utils import get_predictions
from corpus.labels import Entity, Relation, Event
from utils.misc import nest_dict
from config.constants import TRIGGER
from layers.utils import set_model_device, set_tensor_device
from layers.padding import  pad3D, pad2D, pad1D




def get_sent_index(start, end, sent_offsets):

    index = None
    for i, (sent_start, sent_end) in enumerate(sent_offsets):
        if (start >= sent_start) and (start < sent_end):
            index = i
            break

    return index

def get_sent_labels(relations, sent_offsets, subtype_combo, out_type='dict'):

    '''
    Get argument tensors


    Parameters
    ----------


    '''


    assert isinstance(relations, list), str(relations)

    # Sentence count
    sentence_count = len(sent_offsets)

    # initialize output
    sent_labels = [0 for _ in range(sentence_count)]

    # iterate over relations
    for relation in relations:

        entity_a = relation.entity_a
        entity_b = relation.entity_b

        # get indexes of word piece for map bang character spans to span indices
        sent_index_a = get_sent_index( \
                                    start = entity_a.start,
                                    end = entity_a.end,
                                    sent_offsets = sent_offsets)

        sent_index_b = get_sent_index( \
                                    start = entity_b.start,
                                    end = entity_b.end,
                                    sent_offsets = sent_offsets)

        # print()
        # print('a', sent_index_a,entity_a)
        # print('b', sent_index_b,entity_b)

        sent_match = (sent_index_a == sent_index_b) and (sent_index_a is not None)
        subtype_match = (entity_a.subtype, entity_b.subtype) == subtype_combo

        if not sent_match:
            logging.warn("Relation spans multiple sentences: {}".format(relation))

        if sent_match and subtype_match:

            # Save label triple
            assert isinstance(sent_index_a, int), f'{sent_index_a}, {type(sent_index_a)}'
            assert isinstance(sentence_count, int), f'{sentence_count}, {type(sentence_count)}'
            assert sent_index_a < sentence_count, f'{sent_index_a} vs {sentence_count}'

            sent_labels[sent_index_a] = 1


    # for k, v in sent_labels.items():
        # for i, v_ in enumerate(v):
            # if v_:
                # print(k, i, v_)

    if out_type == 'dict':
        pass
    elif out_type == 'list':
        sent_labels = nest_dict(sent_labels)
    else:
        raise ValueError("invalid out_type")

    return sent_labels



def get_sent_labels_multi(relations, sent_offsets, sent_definition, out_type='dict'):



    # iterate over a document label types
    sent_labels = OrderedDict()
    for doc_label, subtype_combos in sent_definition.items():

        # iterate over sentence label combinations
        sent_labels[doc_label] = OrderedDict()
        for combo in subtype_combos:

            # make sure combo is tuple
            if isinstance(combo, list):
                combo = tuple(combo)

            # get sentence sent_labels for specific dock label hyphen subtype combination
            sent_labels[doc_label][combo] = get_sent_labels( \
                        relations = relations,
                        sent_offsets = sent_offsets,
                        subtype_combo = combo,
                        out_type = 'dict')


    n = 0
    for doc_label, subtype_combos in sent_labels.items():
        for combo, values in subtype_combos.items():
            for v in values:
                n += int(v > 0)


    return sent_labels


def tensorfy_doc_labels_multi(doc_labels, device=None):

    tensor_dict = OrderedDict()
    for doc_label, v in doc_labels.items():
        v = torch.LongTensor([v]).squeeze()
        v = set_tensor_device(v, device=device)

        tensor_dict[doc_label] = v

    return tensor_dict

def tensorfy_sent_labels_multi(sent_labels, batch_size, device=None):

    tensor_dict = OrderedDict()
    for doc_label, subtype_combos in sent_labels.items():
        tensor_dict[doc_label] = OrderedDict()
        for combo, v in subtype_combos.items():

            v = pad1D(torch.LongTensor(v), batch_size)
            v = set_tensor_device(v, device=device)

            tensor_dict[doc_label][combo] = v

    return tensor_dict

def decode_document_labels(scores, id2label, as_list=True):

    d = OrderedDict()
    for name, scores_tmp in scores.items():

        batch_size, num_tags = tuple(scores_tmp.shape)

        # get label index
        #index = scores_tmp.max(-1)[1].tolist()
        indices = scores_tmp.max(-1)[1].tolist()

        # get label as string
        #label = id2label[name][index]
        labels = [id2label[name][i] for i in indices]

        #d[name] = label
        d[name] = labels

    # convert to list of dictionary
    if as_list:
        d = nest_dict(d)
    else:
        pass

    return d

def decode_document_prob(scores, id2label, as_list=True):

    d = OrderedDict()
    for name, scores_tmp in scores.items():

        batch_size, num_tags = tuple(scores_tmp.shape)


        labels = [id2label[name][i] for i in range(num_tags)]

        probs = F.softmax(scores_tmp, dim=1).tolist()

        d[name] = []
        for i in range(batch_size):
            P = probs[i]

            assert len(labels) == len(P)
            prob_dict = OrderedDict([(k, v) for k, v in zip(labels, P)])

            d[name].append(prob_dict)

    # convert to list of dictionary
    if as_list:
        d = nest_dict(d)
    else:
        pass

    return d



def decode_sent_labels(scores, seq_mask, as_list=True):

    '''

    Parameters
    ----------
    seq_mask: sequence mask tensor (document count, sentence_count, sequence_length)
    '''


    # (document_count, sentence_count)
    sent_mask = torch.sum(seq_mask, dim=2) > 0
    doc_count, sent_count = tuple(sent_mask.shape)
    sent_mask = sent_mask.tolist()



    d = OrderedDict()
    for doc_label, subtype_combos in scores.items():
        d[doc_label] = OrderedDict()
        for combo, scores_tmp in subtype_combos.items():

            # (document_count, sentence_count)
            indices = scores_tmp.max(-1)[1]
            assert tuple(indices.shape) == (doc_count, sent_count)
            assert indices.min() >= 0, "assuming binary labels"
            assert indices.max() <= 1, "assuming binary labels"

            indices = indices.tolist()

            labels = []
            for ind, mask in zip(indices, sent_mask):
                labs = [i for i, m in zip(ind, mask) if m]
                labels.append(labs)

            d[doc_label][combo] = labels



    # convert to list of dictionary
    if as_list:
        b = []
        for i in range(doc_count):
            c = OrderedDict()
            for doc_label, subtype_combos in d.items():
                c[doc_label] = OrderedDict()
                for combo, labels in subtype_combos.items():
                    c[doc_label][combo] = labels[i]
            b.append(c)
        d = b
    else:
        pass

    return d
