
import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_


from sklearn.metrics import roc_curve, auc


from tensorboardX import SummaryWriter
import pandas as pd
import seaborn as sns

from scipy import interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import joblib
import math
import os
from collections import OrderedDict, Counter

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#from layers.xfmr import tokenize_documents, encode_documents
from layers.xfmr2 import encode_documents, tokenize_documents, wp_tokenize_documents, get_ids_from_sentences
from layers.padding import  pad3D, pad2D, pad1D
from visualization.utils import font_adjust
'''
https://github.com/LightTag/sequence-labeling-with-transformers
https://huggingface.co/transformers/custom_datasets.html#seq-imdb
https://mccormickml.com/2019/07/22/BERT-fine-tuning/


https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/word_att_model.py
'''
from layers.utils import get_predictions, get_label_map
from layers.text_class_utils import get_sent_labels_multi, tensorfy_doc_labels_multi, tensorfy_sent_labels_multi, decode_sent_labels, decode_document_labels, decode_document_prob
from layers.span_utils import SpanMapper
from models.dataset import DatasetBase
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL
from config.constants import ENTITIES, RELATIONS, DOC_LABELS, SENT_LABELS
from layers.utils import set_model_device, set_tensor_device
from corpus.tokenization import tokenize_corpus
from utils.misc import nest_dict

PRETRAINED = "emilyalsentzer/Bio_ClinicalBERT"



# https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
def youdens_j(fpr, tpr, thresholds):
    j_scores = tpr-fpr

    i = np.argmax(j_scores)

    j_opt = j_scores[i]
    fpr_opt = fpr[i]
    tpr_opt = tpr[i]
    threshold_opt = thresholds[i]

    return (j_opt, fpr_opt, tpr_opt, threshold_opt)


def roc(y, y_prob, path, label_type=DOC_LABELS, label=INFILTRATES, \
    figsize = (2.5, 2.5),
    palette = 'tab10',
    alpha = 1.0,
    font_size = 8,
    dpi = 1200
    ):

    assert len(y) == len(y_prob)



    label_names = [list(y_[label_type][label].keys()) for y_ in y_prob]
    probs =  np.array([list(y_[label_type][label].values()) for y_ in y_prob])

    for p in label_names:
        assert label_names[0] == p
    label_names = label_names[0]


    n, num_tags = tuple(probs.shape)
    assert num_tags == 2
    probs = probs[:,1]


    y = [y_[label_type][label] for y_ in y]
    assert len(set(y)) == 2
    y = [label_names.index(y_) for y_ in y]



    fpr, tpr, thresholds = roc_curve(y, probs)
    auc_ = auc(fpr, tpr)

    j_opt, fpr_opt, tpr_opt, threshold_opt = youdens_j(fpr, tpr, thresholds)


    for i in range(1, len(fpr)):
        fpr[i] = min(fpr[i]+i*1e-6, 1.0)
        tpr[i] = min(tpr[i]+i*1e-6, 1.0)


    #f = interpolate.interp1d(fpr, tpr)
    #fpr_new = np.arange(0,1,0.01)
    #tpr_new = f(fpr_new)


    sns.set_theme(style="whitegrid")
    font_adjust(font_size=font_size)


    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, sharey=True)

    g = sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'AUC={auc_:.2f}')

    ax.set_xlabel('FPR', fontweight='bold')
    ax.set_ylabel('TPR', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.05)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc="lower right")


    #ax.tick_params(axis='x', labelrotation=90)

    fn = os.path.join(path, f'roc_{label}.png')
    fig.savefig(fn, dpi=dpi, bbox_inches='tight')


    fn = os.path.join(path, f'roc_orig_{label}.csv')
    df = pd.DataFrame(zip(fpr, tpr), columns=["fpr", "tpr"])
    df.to_csv(fn)

    fn = os.path.join(path, f'youdens_j_{label}.csv')
    a = [(j_opt, fpr_opt, tpr_opt, threshold_opt)]
    b = ['j_opt', 'fpr_opt', 'tpr_opt', 'threshold_opt']
    df = pd.DataFrame(a, columns=b)
    df.to_csv(fn)

    #fn = os.path.join(path, f'roc_interp_{label}.csv')
    #df = pd.DataFrame(zip(fpr_new, tpr_new), columns=["fpr", "tpr"])
    #df.to_csv(fn)


    return df



    #plt.figure()
    #plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel()
    #plt.ylabel()
    #plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    #plt.show()

    #z = sldkjfLDKJF



class DatasetXray(DatasetBase):

    def __init__(self, X, \
            y = None,
            pretrained = PRETRAINED,
            max_length = 30,
            max_wp_length = 50,
            max_sent_count = 20,
            linebreak_bound = True,
            keep = 'last',
            device = None,
            pad_start = True,
            pad_end = True,
            max_span_width = 8,
            min_span_width = 1,
            document_definition = None,
            sent_definition = None,
            entity_definition = None,
            relation_definition = None,
            keep_ws = False
            ):
        super().__init__(X, y)
        '''

        Parameters
        ----------
        X: list of documents as string
        '''

        self.pretrained = pretrained
        self.max_length = max_length
        self.max_wp_length = max_wp_length
        self.max_sent_count = max_sent_count
        self.linebreak_bound = linebreak_bound
        self.keep = keep
        self.device = device
        self.pad_start = pad_start
        self.pad_end = pad_end
        self.max_span_width = max_span_width
        self.min_span_width = min_span_width
        self.sent_definition = sent_definition
        self.keep_ws = keep_ws

        '''
        text processing and embedding
        '''

        self.text = X
        self.X, self.mask, self.offsets = self.preprocess_X(X)

        #self.seq_length = [doc.sum(dim=1) for doc in self.mask]
        self.seq_length = [torch.zeros_like(doc.sum(dim=1)) for doc in self.mask]

        self.document_count = len(self.X)
        self.sent_count = sum([len(doc) for doc in self.X])

        '''
        label processing
        '''
        label_map = OrderedDict()
        label_map.update(document_definition)
        label_map.update(entity_definition)



        self.label2id, self.id2label = get_label_map(document_definition)


        self.span_mapper = SpanMapper( \
                        label_map = label_map,
                        entity_types = list(entity_definition.keys()),
                        relation_combos = relation_definition,
                        batch_size = max_sent_count,
                        max_length = max_length,
                        pad_start = pad_start,
                        pad_end = pad_end,
                        max_span_width = max_span_width,
                        min_span_width = min_span_width,
                        )

        self.y = self.preprocess_y(y)


        logging.info("DataSet X-ray")
        logging.info(f"X, doc count:  {self.document_count}")
        logging.info(f"X, sent count: {self.sent_count}")
        logging.info(f"y, doc count: {None if self.y is None else len(self.y)}")


    def __len__(self):
        return self.document_count


    def __getitem__(self, index):

        doc_index = index


        #seq_length = pad1D(self.seq_length[index], self.max_sent_count)
        #seq_length = set_tensor_device(seq_length, self.device)
        #seq_length = self.seq_length[index]

        # (sentence count, sentence length, embedding dimension)
        seq_tensor = pad3D(self.X[index], self.max_sent_count)
        seq_tensor = set_tensor_device(seq_tensor, self.device)

        # (sentence count, sentence length)
        seq_mask = pad2D(self.mask[index], self.max_sent_count)
        seq_mask = set_tensor_device(seq_mask, self.device)

        # (span_count, 2)
        #span_indices = self.span_mapper.span_indices_tensor(device=self.device)

        # (sentence_count, span_count, 2)
        #span_indices = span_indices.repeat(self.max_sent_count, 1, 1)


        #span_mask = self.span_mapper.span_mask_tensor(seq_length, device=self.device)

        if self.y is None:
            #return (doc_index, seq_tensor, seq_mask, span_indices, span_mask)
            #return (doc_index, seq_tensor, seq_mask, None, None)
            return (doc_index, seq_tensor, seq_mask)
            #y = None
        else:

            y_ = self.y[index]



            y = OrderedDict()
            #y['seq_length'] = seq_length
            #y['span_indices'] = span_indices
            #y['span_mask'] = span_mask
            #y['seq_mask'] = seq_mask

            y["doc_labels"] = tensorfy_doc_labels_multi( \
                                            doc_labels = y_["doc_labels"],
                                            device = self.device)
            y["sent_labels"] = tensorfy_sent_labels_multi( \
                                            sent_labels = y_["sent_labels"],
                                            batch_size = self.max_sent_count,
                                            device = self.device)



            #y["span_labels"] = self.span_mapper.entity_tensor(y_["span_labels"], batch=True, device=self.device)
            #y["role_labels"] = self.span_mapper.relation_tensor(y_["role_labels"], batch=True, device=self.device)





            #return (doc_index, seq_tensor, seq_mask, span_indices, span_mask, y)
            #return (doc_index, seq_tensor, seq_mask, None, None, y)
            return (doc_index, seq_tensor, seq_mask, y)


    def preprocess_X(self, X):
        '''
        Preprocess X
        '''

        #X, mask, offsets = preprocess_X( \
        #                            X = X,
        #                            max_length = self.max_length,
        #                            device = self.device)


        #sentences, sent_offsets, token_offsets = tokenize_documents( \
        #                                documents = X,
        #                                max_length = self.max_length,
        #                                max_sent_count = self.max_sent_count,
        #                                linebreak_bound = self.linebreak_bound,
        #                                pad_start = self.pad_start,
        #                                pad_end = self.pad_end)


        sentences, sent_offsets, token_offsets = tokenize_corpus( \
                                    documents = X,
                                    max_sent_count = self.max_sent_count,
                                    linebreak_bound = self.linebreak_bound,
                                    keep_ws = self.keep_ws)


        for i, doc in enumerate(token_offsets):
            for j, sent in enumerate(doc):
                if self.pad_start:
                    token_offsets[i][j].insert(0, (-1,-1))

                if self.pad_end:
                    token_offsets[i][j].append((-1,-1))

        #encoded_dict, word_pieces_keep = wp_tokenize_documents( \
        #                                sentences = sentences,
        #                                sent_offsets = sent_offsets,
        #                                token_offsets = token_offsets,
        #                                pretrained = self.pretrained,
        #                                max_wp_length = self.max_wp_length,
        #                                keep = self.keep,
        #                                pad_start = self.pad_start,
        #                                pad_end = self.pad_end)



        encoded_dict = get_ids_from_sentences( \
                                sentences = sentences,
                                pretrained = self.pretrained,
                                max_length = self.max_wp_length)

        X, attention_mask = encode_documents( \
                                encoded_dict = encoded_dict,
                                pretrained = self.pretrained,
                                word_pieces_keep = None,
                                device = self.device,
                                train = False,
                                move_to_cpu = True,
                                max_length = None)



        return (X, attention_mask, token_offsets)

    def preprocess_y(self, y):
        '''
        Preprocess y
        '''
        if y is None:
            return None

        labels = []
        assert len(y) == len(self.offsets)
        for y_, offsets_ in zip(y, self.offsets):

            d = OrderedDict()

            d["doc_labels"] = OrderedDict()
            for k, v in y_[DOC_LABELS].items():
                 d["doc_labels"][k] = self.label2id[k][v]


            for k, v in y_[SENT_LABELS].items():
                for j, vals in v.items():
                    assert min(vals) in [0, 1], vals
                    assert max(vals) in [0, 1], vals
            d["sent_labels"] = y_[SENT_LABELS]



            #d["span_labels"] = self.span_mapper.entity_labels(y_[ENTITIES], offsets_)
            #d["role_labels"] = self.span_mapper.relation_labels(y_[RELATIONS], offsets_)



            labels.append(d)

        return labels

    #def postprocess_y(self, doc_indices, seq_mask, doc_scores, sent_scores, span_scores, span_mask, \
    #                    role_scores, role_span_mask, role_indices):
    def postprocess_y(self, doc_indices, seq_mask, doc_scores, sent_scores):

        '''
        Postprocess y
        '''



        doc_indices = doc_indices.tolist()

        n = len(doc_indices)

        doc_labels = decode_document_labels( \
                                    scores = doc_scores,
                                    id2label = self.id2label)
        assert len(doc_labels) == n

        if sent_scores is not None:
            sent_labels = decode_sent_labels(sent_scores, seq_mask)
            assert len(sent_labels) == n

        # iterate over documents
        y = []
        #for di, ds in zip(doc_indices, doc_scores):
        for i in range(n):

            doc_idx = doc_indices[i]

            d = OrderedDict()

            # document labels
            d[DOC_LABELS] = doc_labels[i]

            if sent_scores is not None:
                d[SENT_LABELS] = sent_labels[i]

            # iterate over documents
            if False and (len(span_scores) > 0):
                d[ENTITIES] = self.span_mapper.decode_entity_labels( \
                                            span_scores = span_scores[i],
                                            span_mask = span_mask[i],
                                            offsets = self.offsets[doc_idx],
                                            text = self.text[doc_idx])

                # document relations
                d[RELATIONS] = self.span_mapper.decode_relation_labels( \
                                            span_scores = span_scores[i],
                                            span_mask = span_mask[i],
                                            role_scores = role_scores[i],
                                            role_span_mask = role_span_mask[i],
                                            role_indices = role_indices[i],
                                            offsets = self.offsets[doc_idx],
                                            text = self.text[doc_idx])

            y.append(d)


        return y


    def postprocess_y_prob(self, doc_indices, seq_mask, doc_scores, sent_scores=None):

        '''
        Postprocess y
        '''



        doc_indices = doc_indices.tolist()

        n = len(doc_indices)

        doc_probs = decode_document_prob( \
                                    scores = doc_scores,
                                    id2label = self.id2label)
        assert len(doc_probs) == n

        # iterate over documents
        y = []
        #for di, ds in zip(doc_indices, doc_scores):
        for i in range(n):

            doc_idx = doc_indices[i]

            d = OrderedDict()

            # document labels
            d[DOC_LABELS] = doc_probs[i]

            y.append(d)

        return y


    def input_summary(self,):
        NotImplementedError("not implemented")
