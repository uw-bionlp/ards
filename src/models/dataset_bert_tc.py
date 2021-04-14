
import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from transformers import AutoTokenizer, AutoModel

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
#from layers.xfmr2 import encode_documents, wp_tokenize_documents
from layers.padding import  pad3D, pad2D, pad1D

'''
https://github.com/LightTag/sequence-labeling-with-transformers
https://huggingface.co/transformers/custom_datasets.html#seq-imdb
https://mccormickml.com/2019/07/22/BERT-fine-tuning/


https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/word_att_model.py
'''
from layers.utils import get_predictions, get_label_map
from layers.text_class_utils import get_sent_labels_multi, tensorfy_sent_labels_multi, tensorfy_doc_labels_multi, decode_sent_labels, decode_document_labels
from layers.span_utils import SpanMapper
from models.dataset import DatasetBase
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL
from config.constants import ENTITIES, RELATIONS, DOC_LABELS, SENT_LABELS
from layers.utils import set_model_device, set_tensor_device
from utils.misc import nest_dict
from corpus.sent_boudaries import get_corpus_sent_splits

INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'





class DatasetBertTC(DatasetBase):

    def __init__(self, X, pretrained, \
            y = None,
            max_length = 30,
            max_sent_count = 20,
            linebreak_bound = True,
            keep_ws = False,
            device = None,
            doc_definition = None,
            sent_definition = None
            ):
        super().__init__(X, y)
        '''

        Parameters
        ----------
        X: list of documents as string
        '''



        self.pretrained = pretrained
        self.max_length = max_length
        self.max_sent_count = max_sent_count
        self.linebreak_bound = linebreak_bound
        self.keep_ws = keep_ws
        self.device = device

        self.doc_definition = doc_definition
        self.sent_definition = sent_definition




        sentences, self.input_ids, self.attention_mask = self.preprocess_X(X)




        self.document_count = len(sentences)
        self.sent_count = sum([len(doc) for doc in sentences])


        '''
        label processing
        '''
        self.label2id, self.id2label = get_label_map(doc_definition)


        self.y = self.preprocess_y(y)


        logging.info("DataSet BERT text classifier")
        logging.info(f"X, doc count:  {self.document_count}")
        logging.info(f"X, sent count: {self.sent_count}")
        logging.info(f"y, doc count: {None if self.y is None else len(self.y)}")


    def __len__(self):
        return self.document_count


    def __getitem__(self, index):


        doc_index = index


        input_ids = set_tensor_device(self.input_ids[doc_index], device=self.device)
        attention_mask = set_tensor_device(self.attention_mask[doc_index], device=self.device)


        if self.y is None:
            return (input_ids, attention_mask)
            #y = None
        else:

            doc_labels, sent_labels = self.y[index]


            doc_tensors = tensorfy_doc_labels_multi( \
                                            doc_labels = doc_labels,
                                            device = self.device)


            sent_tensors = tensorfy_sent_labels_multi( \
                                            sent_labels = sent_labels,
                                            batch_size = self.max_sent_count,
                                            device = self.device)

            return (input_ids, attention_mask, doc_tensors, sent_tensors)


    def preprocess_X(self, X):
        '''
        Preprocess X
        '''

        #X, mask, offsets = preprocess_X( \
        #                            X = X,
        #                            max_length = self.max_length,
        #                            device = self.device)




        sentences, sent_offsets = get_corpus_sent_splits( \
                                documents = X,
                                max_sent_count = self.max_sent_count,
                                linebreak_bound = self.linebreak_bound,
                                keep_ws = self.keep_ws)


        tokenizer = AutoTokenizer.from_pretrained(self.pretrained)


        input_ids = []
        attention_mask = []
        for i, sents in enumerate(sentences):
            encoded_dict = tokenizer.batch_encode_plus(sents,
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                max_length = self.max_length,           # Pad & truncate all sentences.
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
                is_split_into_words = False)

            if False: #i == 0:
                logging.info('wp_tokenize_doc')
                for i, ids in enumerate(encoded_dict[INPUT_IDS]):
                    wps = tokenizer.convert_ids_to_tokens(ids)[0:15]
                    logging.info(f"{i}: {wps}....")

            input_ids.append(encoded_dict[INPUT_IDS])
            attention_mask.append(encoded_dict[ATTENTION_MASK])

        return (sentences, input_ids, attention_mask)

    def preprocess_y(self, y):
        '''
        Preprocess y
        '''
        if y is None:
            return None

        labels = []

        for y_ in y:

            d = OrderedDict()

            doc_labels = OrderedDict()
            for k, v in y_[DOC_LABELS].items():
                 doc_labels[k] = self.label2id[k][v]

            sent_labels = y_[SENT_LABELS]
            for k, v in sent_labels.items():
                for j, vals in v.items():
                    assert min(vals) in [0, 1], vals
                    assert max(vals) in [0, 1], vals

            labels.append((doc_labels, sent_labels))

        return labels

    def postprocess_y(self, attention_mask, doc_scores, sent_scores):
        '''
        Postprocess y
        '''

        doc_labels = decode_document_labels( \
                                    scores = doc_scores,
                                    id2label = self.id2label,
                                    as_list = True)


        sent_labels = decode_sent_labels(sent_scores, attention_mask,
                                        as_list = True)

        n = len(doc_labels)
        assert n == len(sent_labels)

        # iterate over documents
        y = []
        for i in range(n):

            d = OrderedDict()
            d[DOC_LABELS] = doc_labels[i]
            d[SENT_LABELS] = sent_labels[i]
            y.append(d)

        return y
