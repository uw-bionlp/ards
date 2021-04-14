import torch
from collections import OrderedDict, Counter
import logging


#from layers.xfmr import char2wordpiece, wordpiece2char
from layers.xfmr2 import char2token_idx, token2char_idx

from layers.padding import  pad3D, pad2D
from layers.utils import set_model_device, set_tensor_device, get_label_map
from layers.utils import get_predictions
from corpus.labels import Entity, Relation, Event
from utils.misc import nest_dict
from config.constants import TRIGGER
from layers.text_class_utils import decode_document_labels


DUMMY_INDICES = (-1, -1)




def import_idx(indices, pad_start=True):

    # Return dummy value if indices None
    if indices is None:
        return DUMMY_INDICES

    return tuple(np.array(indices) + int(pad_start))



def export_idx(indices, pad_start=False):

    # Return dummy value if dummy value
    if tuple(indices) == DUMMY_INDICES:
        return DUMMY_INDICES

    # Separate start and end indices
    start, end = tuple(indices)

    # Account for start of sequence padding
    start = start - int(pad_start)
    end = end - int(pad_start)

    return (start, end)




def entity2span_index(entity, offsets, span_map, max_span_width):

    # get indexes of word piece for map bang character spans to span indices
    sent_index, start_index, end_index = char2token_idx( \
                                start = entity.start,
                                end = entity.end,
                                offsets = offsets)

    if sent_index is None:
        return (None, None)
    else:

        if (start_index, end_index) not in span_map:
            end_index_old = end_index
            end_index = start_index + max_span_width
            logging.warn(f"Truncating entity span:")
            logging.warn(f"\tSpan text: {repr(entity.text)}")
            logging.warn(f"\tOrig token indices:  ({start_index},{end_index_old})")
            logging.warn(f"\tTrunc token indices: ({start_index},{end_index})")

        # Index of span
        span_index = span_map[(start_index, end_index)]

        return (sent_index, span_index)



def entity_labels(entities, offsets, entity_types, span_map, label2id, max_span_width, out_type='dict'):
    '''
    Get tensor representation for mention (e.g. trigger)

    Parameters
    ----------

    events: document events as list of list of Event
    span_map: dictionary for mapping span token indices to span index


    label_map: dictionary for mapping labels to label indices
    pad_start: boolean indicating whether token indices should be incremented to account for start of sequence padding

    Returns
    -------

    labels: list of span labels as tuple (span idx, label)

    '''

    assert isinstance(entities, list), str(entities)

    # go to dimensions of input
    sentence_count = len(offsets)

    # initialize output
    labels = OrderedDict()
    for type_ in entity_types:
        labels[type_] = [[] for _ in range(sentence_count)]

    # iterate over entities
    for entity in entities:

        # get sentence and span index
        sent_index, span_index = entity2span_index(entity, offsets, span_map, max_span_width)

        if sent_index is None:
            logging.warn(f"Could not map entity: {entity}")


        else:

            # get label id
            lab_id = label2id[entity.type_][entity.subtype]

            # Include span label
            assert sent_index in range(sentence_count)
            labels[entity.type_][sent_index].append((span_index, lab_id))

    if out_type == 'dict':
        pass
    elif out_type == 'list':
        labels = nest_dict(labels)
    else:
        raise ValueError("invalid out_type")

    return labels


def relation_labels(relations, offsets, relation_combos, span_map, max_span_width, out_type='dict'):

    '''
    Get argument tensors


    Parameters
    ----------

    events: document of events, list of list of Event
    arg_type: string indicating argument type (e.g.'Trigger' or 'Status')
        num_trig: int indicating maximum number of trigger span_indices
    num_arg: int indicating maximum number of argument spans
    arg_map: dictionary for mapping argument labels to indices
    span_map: dictionary for mapping span labels to indices
    label_type: string indicating how to represent argument label (e.g. event type, span type, etc.)


    Returns
    -------

    arg_labels: tensor of trigger-argument labels with shape (num_trig, num_arg)
    mask: tensor of trigger-argument integer mask with shape (num_trig, num_arg)
    span_labels: tensor of argument labels with shape (num_arg)
    span_indices: tensor of mention token indices with shape (num_arg, 2)
            NOTE that span indices are Inclusive for Allen NLP

    '''

    assert isinstance(relations, list), str(relations)

    # Sentence count
    sentence_count = len(offsets)

    # initialize output
    labels = OrderedDict()
    for combos in relation_combos:
        if isinstance(combos, list):
            combos = tuple(combos)
        labels[combos] = [[] for _ in range(sentence_count)]

    # iterate over relations
    for relation in relations:

        entity_a = relation.entity_a
        entity_b = relation.entity_b

        # get sentence and span index
        sent_index_a, span_index_a = entity2span_index(entity_a, offsets, span_map, max_span_width)
        sent_index_b, span_index_b = entity2span_index(entity_b, offsets, span_map, max_span_width)

        if (sent_index_a == sent_index_b) and (sent_index_a is not None):

            # Save label triple
            assert isinstance(sent_index_a, int), f'{sent_index_a}, {type(sent_index_a)}'
            assert isinstance(sentence_count, int), f'{sentence_count}, {type(sentence_count)}'
            assert sent_index_a < sentence_count, f'{sent_index_a} vs {sentence_count}'

            combo = (entity_a.type_, entity_b.type_)

            labels[combo][sent_index_a].append((span_index_a, span_index_b, 1))
        else:

            logging.warn("Relation spans multiple sentences: {}".format(relation))

    if out_type == 'dict':
        pass
    elif out_type == 'list':
        labels = nest_dict(labels)
    else:
        raise ValueError("invalid out_type")

    return labels



def enumerate_span_indices(start, max_seq_length, min_span_width, max_span_width):

    # Iterate over token positions from start through max length
    span_indices = []
    for i in range(start, max_seq_length):

        # End indices to traverse
        first_end_index = min(i + min_span_width - 1, max_seq_length)
        last_end_index =  min(i + max_span_width,     max_seq_length)

        # Loop on start-end combinations
        for j in range(first_end_index, last_end_index):
            span_indices.append((i, j + 1))

    return span_indices


def get_span_mask(max_seq_length, span_indices):

    # Create span mask for all possible sequence lengths
    span_mask = OrderedDict()
    for seq_len in range(max_seq_length + 2):
        span_mask[seq_len] = [int(i <= seq_len) for _, i in span_indices]

    return span_mask


def label_tensor(label):

    return torch.LongTensor([label]).squeeze()



def entity_tensor_seq(entities, span_count):


    # iterate over entity types
    tensor_dict = OrderedDict()
    for type_, labels in entities.items():

        # initialize output tensor
        tensor_dict[type_] = torch.zeros(span_count, dtype=torch.long)

        # Loop on labels
        for span_index, lab in labels:
            tensor_dict[type_][span_index] = lab

    return tensor_dict


def entity_tensor_batch(entity_dict, span_count, seq_count):

    # iterate over entity types
    tensor_dict = OrderedDict()
    for type_, entity_batch in entity_dict.items():

        # initialize output tensor
        tensor_dict[type_] = torch.zeros(seq_count, span_count, dtype=torch.long)

        # iterate over sentences
        for seq_index, entity_seq in enumerate(entity_batch):

            # iterate over labels in sentence
            for span_index, lab in entity_seq:

                if seq_index < seq_count:
                    tensor_dict[type_][seq_index][span_index] = lab


        #tensor_dict[type_] = torch.zeros(seq_count, span_count, dtype=torch.long) + 1

    return tensor_dict


def relation_tensor_seq(relation_dict, span_count):

    # iterate over entity types
    tensor_dict = OrderedDict()
    for type_, relations in relation_dict.items():

        # initialize output tensor
        tensor = torch.zeros(span_count, span_count, dtype=torch.long)

        # Loop on labels
        for index_a, index_b, lab in relations:
            tensor[index_a][index_b] = lab

        tensor_dict[type_] = tensor

    return tensor_dict


def relation_tensor_batch(relation_dict, span_count, seq_count):

    # iterate over entity types
    tensor_dict = OrderedDict()
    for type_, relations in relation_dict.items():

        # Initialize output tensor
        tensor = torch.zeros(seq_count, span_count, span_count, dtype=torch.long)

        # iterate over sentences
        for seq_index, relation_seq in enumerate(relations):

            # loop on labels in sentence
            for index_a, index_b, lab in relation_seq:

                if seq_index < seq_count:
                    tensor[seq_index][index_a][index_b] = lab

        tensor_dict[type_] = tensor

    return tensor_dict

'''
def tensorfy_Y(self, d, seq_len):



    #assert len(labels) == len(self.seq_lengths)
    #tensors = []
    #for d, seq_len in zip(labels, self.seq_lengths):


    g = OrderedDict()

    # Span indices as tensor
    # (span_count, 2)
    g['span_indices'] = torch.LongTensor(self.span_indices)
    #g['span_midpoint'] = torch.round(torch.mean(torch.FloatTensor(self.span_indices), dim=-1)
    g['span_overlaps'] = torch.BoolTensor(self.span_overlaps)
    g['span_mask'] = torch.LongTensor(self.span_mask[seq_len])
    g['seq_length'] =  torch.LongTensor([seq_len]).squeeze()



    mention_labels = OrderedDict()
    for name, labels in d['mention_labels'].items():
        mention_labels[name] = get_mention_tensor(labels, self.span_count)

    indicator_labels = OrderedDict()
    for name, labels in d['indicator_labels'].items():
        indicator_labels[name] = torch.LongTensor(labels)

    indicator_weights = OrderedDict()
    for name, labels in d['indicator_weights'].items():
        indicator_weights[name] = torch.FloatTensor(labels)

    seq_labels = OrderedDict()
    for name, labels in d['seq_labels'].items():
        seq_labels[name] = torch.LongTensor(labels)


    arg_labels = OrderedDict()
    for name, labels in d['arg_labels'].items():
        arg_labels[name] = get_arg_tensor(labels, self.span_count)

    g['mention_labels'] = mention_labels
    g['indicator_labels'] = indicator_labels
    g['indicator_weights'] = indicator_weights
    g['seq_labels'] = seq_labels
    g['arg_labels'] = arg_labels

    return g
'''




def decode_entity_labels(span_scores, span_mask, span_indices, offsets, id2label, text=None, out_type='list'):

    '''

    Parameters
    ----------

    span_scores: dictionary of span score tensors (batch_size, span_count, label_count)
    span_mask: tensor mask (batch_size, span_count)
    span_indices: maps spans to word indices. list of span indice tuples, e.g. [(1,2), (1,3)...(38,39)], len=span_count
    offsets: maps words to character indices. list of list of char indices for words

    '''

    d = OrderedDict()
    entities = OrderedDict()
    for name, scores in span_scores.items():

        batch_size, span_count, label_count = tuple(scores.shape)

        # (batch_size, span_count)
        label_indices = get_predictions(scores, span_mask)

        label_indices = label_indices.tolist()


        # iterate over sequences in batch
        for i_seq, seq in enumerate(label_indices):

            # iterate over spans in sequence
            for i_span, label_index in enumerate(seq):

                # only create entities for non-null labels
                if label_index > 0:

                    # get label from index
                    label = id2label[name][label_index]

                    # get word indices from span indices
                    start_index, end_index = span_indices[i_span]

                    assert i_seq < len(offsets), f'{offsets} -- {i_seq}'

                    # get character indices from word indices
                    #start, end = wordpiece2char( \
                    #            start = word_indices[0],
                    #            end = word_indices[1],
                    #            offsets = offsets[i_seq])

                    start, end = token2char_idx( \
                                sent_index = i_seq,
                                start_index = start_index,
                                end_index = end_index,
                                offsets = offsets)

                    if text is None:
                        txt = None
                    else:
                        txt = text[i_seq][start:end]

                    entity = Entity( \
                                    type_ = name,
                                    start = start,
                                    end = end,
                                    text = txt,
                                    subtype = label,
                                    sent_index = i_seq)

                    entities[(name, i_seq, i_span)] = entity


    if out_type == 'list':
        entities = [v for k, v in entities.items()]
    elif out_type == 'dict':
        pass
    else:
        raise ValueError("invalid out_type")

    return entities





def decode_relation_labels(span_scores, span_mask, span_indices, role_scores, \
    role_span_mask, role_indices, offsets, id2label, text=None, out_type='list'):

    '''

    Parameters
    ----------

    span_scores: dictionary of span score tensors (batch_size, span_count, label_count)
    span_mask: tensor mask (batch_size, span_count)
    span_indices: maps spans to word indices. list of span indice tuples, e.g. [(1,2), (1,3)...(38,39)], len=span_count


    role_scores: dict of tensor (batch_size, top_span_count, top_span_count, 2)
    role_span_mask:  tensor (batch_size, top_span_count)
    role_indices: tensor (batch_size, top_span_count)


    offsets: maps words to character indices. list of list of char indices for words

    '''

    entities = decode_entity_labels( \
                            span_scores = span_scores,
                            span_mask = span_mask,
                            span_indices = span_indices,
                            offsets = offsets,
                            id2label = id2label,
                            text = text,
                            out_type = 'dict')


    for (a, b), scores in role_scores.items():

        # labels (batch_size, span_count_a, span_count_b)
        batch_size, span_count_a, span_count_b, label_count = tuple(scores.shape)

        if span_count_a != span_count_b:
            NotImplementedError("Currently assumes span counts are the same, need to update to make more general")

        batch_size, top_span_count = tuple(role_span_mask.shape)
        assert top_span_count == span_count_a
        assert label_count == 2

        mask_a = role_span_mask.unsqueeze(1).repeat(1, span_count_a, 1)
        mask_b = role_span_mask.unsqueeze(2).repeat(1, 1, span_count_b)
        role_mask = mask_a * mask_b

        # (batch_size, top_span_count, top_span_count)
        label_indices = get_predictions(scores, role_mask)
        label_indices = label_indices.tolist()

        # iterate over sequences in batch
        relations = []
        for i_seq, sequence in enumerate(label_indices):

            # iterate over spans in sequence
            for i_span_a, spans_a in enumerate(sequence):

                # iterate over spans in sequence
                for i_span_b, label_index in enumerate(spans_a):

                    if label_index > 0:


                        entity_a = None
                        i_span = role_indices[i_seq][i_span_a]
                        k = (a, i_seq, i_span)
                        if k in entities:
                            entity_a = entities[k]


                        entity_b = None
                        i_span = role_indices[i_seq][i_span_b]
                        k = (b, i_seq, i_span)
                        if k in entities:
                            entity_b = entities[k]



                        if (entity_a is not None) and (entity_b is not None):
                            relation = Relation( \
                                entity_a,
                                entity_b,
                                role = label_index)

                            relations.append(relation)


    return relations


class SpanMapper(object):

    '''
    Used code and approach from AllenNLP
        https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_utils/span_utils.py

    NOTE: span end indices are EXCLUSIVE
    '''

    def __init__(self, \
                label_map = None,
                label2id = None,
                id2label = None,
                entity_types = None,
                relation_combos = None,
                batch_size = None,
                max_length = 30,
                pad_start = True,
                pad_end = True,
                max_span_width = 8,
                min_span_width = 1,
                ):

        cnt = 2

        logging.info("="*80)
        logging.info("SpanMapper")
        logging.info("="*80)


        if label_map is None:
            assert label2id is not None
            assert id2label is not None
            self.label2id = label2id
            self.id2label = id2label
        else:
            self.label2id, self.id2label = get_label_map(label_map)

        self.entity_types = entity_types
        self.relation_combos = relation_combos

        # ignore first token index, if start padded
        self.pad_start = pad_start
        self.start = int(pad_start)
        logging.info("start (adjusted): {}".format(self.start))

        # ignore last token index if end padded
        self.pad_end = pad_end
        self.max_length = max_length - int(pad_end)
        logging.info("max_length (adjusted): {}".format(self.max_length))

        # maximum sequence count
        self.batch_size = batch_size

        # enumerate all possible span indices
        self.max_span_width = max_span_width
        self.min_span_width = min_span_width
        self.span_indices = enumerate_span_indices( \
                    start = self.start,
                    max_seq_length = self.max_length,
                    min_span_width = self.min_span_width,
                    max_span_width = self.max_span_width)
        logging.info('minimum span index:\t{}'.format(min([s for s, e in self.span_indices])))
        logging.info('maximum span index:\t{}'.format(max([e for s, e in self.span_indices])))
        logging.info('span examples:\n{}{}{}'.format(self.span_indices[0:cnt], '.'*10, self.span_indices[-cnt:]))

        # number of spans for maximum sentence length
        self.span_count = len(self.span_indices)

        # mapping from span token indices to span index
        self.span_map = OrderedDict((s, i) for i, s in enumerate(self.span_indices))
        logging.info('span map:')
        for j, (s, i) in enumerate(self.span_map.items()):
            if (j < cnt) or (self.span_count - j < cnt):
                #pass
                logging.info('\t{} --> {}'.format(s, i))

        # create span mask given the sequence length
        self.span_mask = get_span_mask(self.max_length, self.span_indices)
        logging.info('span mask:')
        for j, (len_, msk) in enumerate(self.span_mask.items()):
            if (j < cnt) or (len(self.span_mask) - j < cnt):
                pass
                #logging.info('\tseq len={} --> mask={}'.format(len_, msk))



    def map_label(self, type_, subtype):
        return self.label2id[type_][subtype]

    def document_labels(self, labels):

        labels = OrderedDict([(type_, self.label2id[type_][subtype]) \
                                        for type_, subtype in labels.items()])

        return labels

    def entity_labels(self, entities, offsets, out_type='dict'):

        labels = entity_labels( \
                        entities = entities,
                        offsets = offsets,
                        entity_types = self.entity_types,
                        span_map = self.span_map,
                        label2id = self.label2id,
                        max_span_width = self.max_span_width,
                        out_type = out_type)

        return labels

    def relation_labels(self, relations, offsets, out_type="dict"):

        labels = relation_labels( \
                        relations = relations,
                        offsets = offsets,
                        relation_combos = self.relation_combos,
                        span_map = self.span_map,
                        max_span_width = self.max_span_width,
                        out_type = out_type)

        return labels

    def label_tensor(self, label):

        tensor = label_tensor(label)

        return tensor

    def document_tensor(self, labels, device=None):

        tensor_dict = OrderedDict([(type_, label_tensor(label)) \
                                        for type_, label in labels.items()])

        for k, v in tensor_dict.items():
            tensor_dict[k] = set_tensor_device(tensor_dict[k], device)

        return tensor_dict

    def entity_tensor(self, entities, batch=False, device=None):

        if batch:
            tensor_dict = entity_tensor_batch(entities, self.span_count, self.batch_size)
        else:
            tensor_dict = entity_tensor_seq(entities, self.span_count)

        for k, v in tensor_dict.items():
            tensor_dict[k] = set_tensor_device(v, device)

        return tensor_dict

    def relation_tensor(self, relations, batch=False, device=None):

        if batch:
            tensor_dict = relation_tensor_batch(relations, self.span_count, self.batch_size)
        else:
            tensor_dict = relation_tensor_seq(relations, self.span_count)

        for k, v in tensor_dict.items():
            tensor_dict[k] = set_tensor_device(v, device)

        return tensor_dict

    def span_indices_tensor(self, device=None):

        tensor = torch.LongTensor(self.span_indices)

        tensor = set_tensor_device(tensor, device)

        return tensor

    def span_mask_tensor(self, seq_length, device=None):

        if torch.is_tensor(seq_length):
            seq_length = seq_length.tolist()

        if isinstance(seq_length, (int, float)):
            tensor = torch.LongTensor(self.span_mask[seq_length])
        else:
            tensor = torch.LongTensor([self.span_mask[n] for n in seq_length])

        tensor = set_tensor_device(tensor, device)

        return tensor

    def decode_document_labels(self, scores):

        NotImplementedError("need to update, function likely changed to accommodate patched documents")
        labels = decode_document_labels(scores, self.id2label)

        return labels

    def decode_entity_labels(self, span_scores, span_mask, offsets, text=None):

        entities = decode_entity_labels( \
                                span_scores = span_scores,
                                span_mask = span_mask,
                                span_indices = self.span_indices,
                                offsets = offsets,
                                id2label = self.id2label,
                                text = text)

        return entities

    def decode_relation_labels(self, span_scores, span_mask, role_scores, \
                        role_span_mask, role_indices, offsets, text=None):

        relations = decode_relation_labels( \
                                span_scores = span_scores,
                                span_mask = span_mask,
                                span_indices = self.span_indices,
                                role_scores = role_scores, \
                                role_span_mask = role_span_mask,
                                role_indices = role_indices,
                                offsets = offsets,
                                id2label = self.id2label,
                                text = text)


        return relations
