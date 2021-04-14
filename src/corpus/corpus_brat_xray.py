

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from collections import OrderedDict, Counter
import hashlib
import logging

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



from corpus.corpus_brat import CorpusBrat, counter2df
from corpus.document_brat_xray import DocumentBratXray
from config.constants import TRAIN, DEV, TEST
from config.constants_pulmonary import FOOTER, INFILTRATES, EXTRAPARENCHYMAL
from config.constants import ENTITIES, RELATIONS, DOC_LABELS, SENT_LABELS
from layers.xfmr2 import tokenize_documents
from layers.text_class_utils import get_sent_labels_multi

def id2tags(id):

    X = id.split(os.sep)
    round, annotator, stem = tuple(X)
    tags = set([round, annotator])


    return tags

class CorpusBratXray(CorpusBrat):

    def __init__(self):

        CorpusBrat.__init__(self, document_object=DocumentBratXray)


    # OVERRIDE
    def label_summary(self, path=None, **kwargs):

        counter_entities_sub = Counter()
        counter_entities_no_sub = Counter()
        counter_relations = Counter()
        for doc in self.docs(out_type='list', **kwargs):
            c_entities_sub, c_entities_no_sub, c_relations = doc.label_summary()
            counter_entities_sub += c_entities_sub
            counter_entities_no_sub += c_entities_no_sub
            counter_relations += c_relations

        counters =  [ \
            ('entities_match_subtype', counter_entities_sub),
            ('entities_no_subtype_match', counter_entities_no_sub),
            ('relations', counter_relations)
            ]
        dfs = OrderedDict()
        for name, counter in counters:
            df = counter2df(counter)
            dfs[name] = df

            if path is not None:
                f = os.path.join(path, "label_summary_{}.csv".format(name))
                df.to_csv(f)

        return dfs

    # OVERRIDE
    def y(self, doc_map=None, side_swap=False, **kwargs):
        y = []
        for doc in self.docs(**kwargs):
            y.append(doc.y(doc_map=doc_map, side_swap=side_swap))
        return y


    # OVERRIDE
    def Xy(self, doc_map=None, side_swap=False, **kwargs):
        #X = []
        #y = []
        #for doc in self.docs(**kwargs):
        #
        #    X_, y_ = doc.Xy(doc_map=doc_map, side_swap=side_swap, tokenization_params=tokenization_params, sent_definition=sent_definition)
        #    X.append(X_)
        #    y.append(y_)

        X = self.X(**kwargs)
        y = self.y(doc_map=doc_map, side_swap=side_swap, **kwargs)

        return (X, y)
