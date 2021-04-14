
import torch
import torch.nn.functional as F
from collections import OrderedDict, Counter
import numpy as np

def set_model_device(x, device=None):

    if device is not None:
        device = torch.device('cuda:{}'.format(device))
        x.to(device)


def set_tensor_device(x, device=None):

    if device is not None:
        device = torch.device('cuda:{}'.format(device))
        x = x.to(device)

    return x



def get_loss(labels, scores, mask, reduction='sum', weight=None):
    '''
    Calculate loss for arguments (e.g. trigger-entity pairs)

    Parameters
    ----------
    scores: (batch_size, trig_num, entity_num, tag_num)
    labels: (batch_size, trig_num, entity_num)
    mask: (batch_size, trig_num, entity_num)

    Returns
    -------
    loss: scalar
    '''

    # Flatten fields
    # (batch_size*trig_num*entity_num,  tag_num)
    tag_num = scores.size(-1)
    scores_flat = scores.view(-1, tag_num)
    # (batch_size*trig_num*entity_num)
    labels_flat = labels.view(-1)
    # (batch_size*trig_num*entity_num)
    mask_flat = mask.view(-1).bool()

    # Loss
    loss = F.cross_entropy(scores_flat[mask_flat],
                           labels_flat[mask_flat],
                           reduction = reduction,
                           weight = weight)
    return loss

def aggregate(x, reduction='sum'):
    if reduction == 'sum':
        x = torch.sum(x)
    elif reduction == 'mean':
        x = torch.mean(x)
    else:
        ValueError(f"invalid reduction: {reduction}")

    return x



def get_predictions(scores, mask):
    '''
    Get predictions from logits and apply mask

    Parameters
    ----------
    scores: (batch_size, num_trig, num_entity, num_tags) OR
                        (batch_size, num_entity, num_tags)
    mask: (batch_size, num_trig, num_entity) OR
                        (batch_size, num_entity)
    '''

    # Get predictions from label scores
    # (batch_size, num_trig, num_entity) OR (batch_size, num_entity)
    max_scores, label_indices = scores.max(-1)

    # For masked spans to NULL label (i.e. 0)
    labels_masked = label_indices.to(mask.device)*mask

    return labels_masked



def default_counts_tensor():
    return torch.tensor([0.0, 0.0, 0.0])

def perf_counts(labels, scores, mask=None, as_tensor=True):

    if mask is None:
        mask = torch.ones_like(labels)

    # get predictions
    predictions = scores.max(-1)[1]

    # mask for only positive labels
    labels_positive =      ((labels      > 0)*mask).float()
    predictions_positive = ((predictions > 0)*mask).float()

    # labels and predictions equal
    equal = ((labels == predictions)*mask).float()

    # true positives
    tp = (equal*labels_positive).sum()

    # number of positive labels in truth
    nt = labels_positive.sum()

    # number of positive labels in prediction
    np = predictions_positive.sum()

    if as_tensor:
        return torch.tensor([tp, nt, np])
    else:
        return (tp, nt, np)


def perf_counts_multi(label_dict, score_dict, mask=None, as_tensor=True):

    counts = []
    for k in label_dict:
        labels = label_dict[k]
        scores = score_dict[k]
        counts.append(perf_counts(labels, scores, mask, as_tensor=True))
    counts = torch.stack(counts, dim=0).sum(dim=0)

    if as_tensor:
        return counts
    else:
        return tupl(counts)


def prf_from_counts(tp, nt, np, as_tensor=True):

    # precision
    if np:
        p = tp/np
    else:
        p = 0

    # recall
    if nt:
        r = tp/nt
    else:
        r = 0

    # f1 score
    if p + r:
        f = 2*p*r/(p+r)
    else:
        f = 0

    if as_tensor:
        return torch.tensor([p, r, f])
    else:
        return (p, r, f)

def PRF1(labels, scores, mask=None, as_tensor=True):

    tp, nt, np = perf_counts(labels, scores, mask, as_tensor=False)

    x = prf_from_counts(tp, nt, np, as_tensor=as_tensor)

    return x



def PRF1multi(label_dict, score_dict, mask=None, as_tensor=True):


    counts = []
    for k in label_dict:
        labels = label_dict[k]
        scores = score_dict[k]

        counts.append(perf_counts(labels, scores, mask, as_tensor=True))

    counts = torch.stack(counts, dim=0).sum(dim=0)

    x = prf_from_counts(*tuple(counts), as_tensor=as_tensor)

    return x


class PRFAggregator():

    def __init__(self):
        self.counts = OrderedDict()

    def update_counts(self, new_counts):

        for k, v in new_counts.items():
            v = v.cpu().numpy()
            v = np.nan_to_num(v)

            if k not in self.counts:
                self.counts[k] = v
            else:
                self.counts[k] += v

        return True


    def prf(self, as_string=True):


        prf = OrderedDict()
        for k, v in self.counts.items():
            prf[k] = prf_from_counts(*tuple(v), as_tensor=False)

        if as_string:
            out = []
            for k, (p, r, f) in prf.items():
                out.append(f'{k}={p:.2f}/{r:.2f}/{f:.2f}')
            return ', '.join(out)
        else:
            return prf

    def reset(self):
        self.counts = OrderedDict()



def get_doc_index_map(corpus):

    k = 0
    to_sent_index = OrderedDict()
    from_sent_index = OrderedDict()
    for i, doc in enumerate(corpus):
        for j, sent in enumerate(doc):
            assert (i, j) not in to_sent_index
            to_sent_index[(i, j)] = k

            assert k not in from_sent_index
            from_sent_index[k] = (i, j)

            k += 1

    return (to_sent_index, from_sent_index)



def get_label_map(labels, verbose=False):

    to_id = OrderedDict()
    to_lab = OrderedDict()
    for type_, labs in labels.items():
        to_id[type_] = OrderedDict()
        to_lab[type_] = OrderedDict()
        for i, lab in enumerate(labs):
            to_id[type_][lab] = i
            to_lab[type_][i] = lab

    if verbose:
        logging.info('-'*72)
        logging.info('Label-ID mapping:')
        logging.info('-'*72)
        logging.info('Label to ID map:')
        for name, map_ in to_id.items():
            logging.info('')
            logging.info(name)
            for k, v in map_.items():
                logging.info('{} --> {}'.format(k, v))

        logging.info('ID to Label map:')
        for name, map_ in to_lab.items():
            logging.info('')
            logging.info(name)
            for k, v in map_.items():
                logging.info('{} --> {}'.format(k, v))

    return (to_id, to_lab)
