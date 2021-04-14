
import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_


from tensorboardX import SummaryWriter

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
from layers.xfmr2 import encode_documents, tokenize_documents
from layers.padding import  pad3D, pad2D, pad1D


from layers.utils import get_predictions, get_doc_index_map
from layers.span_utils import SpanMapper
from models.dataset import DatasetBase
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL
from config.constants import ENTITIES, RELATIONS, DOC_LABELS, EVENTS
from layers.utils import set_model_device, set_tensor_device
from corpus.tokenization import get_tokenizer
from corpus.labels import events2relations



PRETRAINED = "emilyalsentzer/Bio_ClinicalBERT"



def flatten(X):
    return [x_ for x in X for x_ in x]


class DatasetSymptoms(DatasetBase):

    def __init__(self, X, \
            y = None,
            pretrained = PRETRAINED,
            max_length = 30,
            max_wp_length = 50,
            linebreak_bound = True,
            keep = 'last',
            device = None,
            pad_start = True,
            pad_end = True,
            max_span_width = 8,
            min_span_width = 1,
            entity_definition = None,
            relation_definition = None,
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
        self.linebreak_bound = linebreak_bound
        self.keep = keep
        self.device = device
        self.pad_start = pad_start
        self.pad_end = pad_end
        self.max_span_width = max_span_width
        self.min_span_width = min_span_width

        '''
        text processing and embedding
        '''

        self.text = X
        self.X, self.mask, self.offsets = self.preprocess_X(X)

        self.doc_sent2index, self.index2doc_sent = get_doc_index_map(self.X)
        self.sent_count = len(self.index2doc_sent)

        self.seq_length = [doc.sum(dim=1) for doc in self.mask]
        self.document_count = len(self.X)

        '''
        label processing
        '''

        self.span_mapper = SpanMapper( \
                        label_map = entity_definition,
                        entity_types = list(entity_definition.keys()),
                        relation_combos = relation_definition,
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
        return self.sent_count


    def __getitem__(self, index):

        device = self.device


        doc_index, sent_index = self.index2doc_sent[index]


        seq_length =self.seq_length[doc_index][sent_index]
        seq_length = set_tensor_device(seq_length, device)


        # (sentence length, embedding dimension)
        seq_tensor = self.X[doc_index][sent_index]
        seq_tensor = set_tensor_device(seq_tensor, device)

        # (sentence length)
        seq_mask = self.mask[doc_index][sent_index]
        seq_mask = set_tensor_device(seq_mask, device)

        span_indices = self.span_mapper.span_indices_tensor(device=device)
        span_mask = self.span_mapper.span_mask_tensor(seq_length, device=device)

        if self.y is None:
            return (index, seq_tensor, seq_mask, span_indices, span_mask)
            #y = None
        else:

            y_ = self.y[doc_index][sent_index]

            y = OrderedDict()
            y['span_indices'] = span_indices
            y['span_mask'] = span_mask
            #y['seq_mask'] = seq_mask
            y["span_labels"] = self.span_mapper.entity_tensor(y_["span_labels"], batch=False, device=device)
            y["role_labels"] = self.span_mapper.relation_tensor(y_["role_labels"], batch=False, device=device)

            return (index, seq_tensor, seq_mask, span_indices, span_mask, y)


    def preprocess_X(self, X):
        '''
        Preprocess X
        '''

        #X, mask, offsets = preprocess_X( \
        #                            X = X,
        #                            max_length = self.max_length,
        #                            device = self.device)

        token_offsets, encoded_dict, word_pieces_keep = tokenize_documents( \
                                        documents = X,
                                        pretrained = self.pretrained,
                                        max_length = self.max_length,
                                        max_wp_length = self.max_wp_length,
                                        linebreak_bound = self.linebreak_bound,
                                        keep_ws = False,
                                        keep = self.keep,
                                        pad_start = self.pad_start,
                                        pad_end = self.pad_end)


        X, mask = encode_documents( \
                                        encoded_dict = encoded_dict,
                                        word_pieces_keep = word_pieces_keep,
                                        pretrained = self.pretrained,
                                        device = self.device,
                                        train = False,
                                        detach = True,
                                        move_to_cpu = True,
                                        max_length = self.max_length)

        return (X, mask, token_offsets)

    def preprocess_y(self, y):
        '''
        Preprocess y
        '''
        if y is None:
            return None

        labels = []
        assert len(y) == len(self.offsets)
        for y_, offsets_ in zip(y, self.offsets):

            entities = y_[ENTITIES]
            relations = events2relations(y_[EVENTS])

            span_labels = self.span_mapper.entity_labels(entities, offsets_, out_type='list')
            role_labels = self.span_mapper.relation_labels(relations, offsets_, out_type='list')

            assert len(span_labels) == len(role_labels)
            labs = []
            for s, r in zip(span_labels, role_labels):
                d = OrderedDict()
                d["span_labels"] = s
                d["role_labels"] = r
                labs.append(d)
            labels.append(labs)

        return labels

    def postprocess_y(self, indices, span_scores, span_mask, \
                        role_scores, role_span_mask, role_indices):
        '''
        Postprocess y
        '''


        assert len(doc_scores) == len(doc_indices)
        if len(span_scores) > 0:
            assert len(doc_scores) == len(span_scores)
            assert len(doc_scores) == len(span_mask)



        doc_indices = doc_indices.tolist()

        n = len(doc_scores)

        # iterate over documents
        y1 = []
        #for di, ds in zip(doc_indices, doc_scores):
        for i in range(n):

            d = OrderedDict()

            # document labels
            d[DOC_LABELS] = self.span_mapper.decode_document_labels( \
                                                scores = doc_scores[i])
            y1.append(d)


        # iterate over documents
        if len(span_scores) > 0:
            y2 = []
            for i in range(n):

                d = OrderedDict()

                doc_idx = doc_indices[i]

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

                y2.append(d)
        else:
            y2 = []
            for _ in range(0, len(doc_scores)):
                d = OrderedDict()
                d[ENTITIES] = []
                d[RELATIONS] = []
                y2.append(d)

        assert len(y1) == len(y2)
        y = []
        for d1, d2 in zip(y1, y2):
            d = OrderedDict()
            d.update(d1)
            d.update(d2)
            y.append(d)

        return y


    def input_summary(self,):
        NotImplementedError("not implemented")
