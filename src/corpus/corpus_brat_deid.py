

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
from corpus.document_brat_deid import DocumentBratDeid

class CorpusBratDeid(CorpusBrat):

    def __init__(self):

        CorpusBrat.__init__(self, document_object=DocumentBratDeid)


    # OVERRIDE
    def label_summary(self, path=None, **kwargs):

        counter_subtype = Counter()
        counter_text = Counter()

        for doc in self.docs(out_type='list', **kwargs):
            c_subtype, c_text = doc.label_summary()
            counter_subtype += c_subtype
            counter_text += c_text

        counters =  [ \
            ('subtype', counter_subtype),
            ('text', counter_text)
            ]
        dfs = OrderedDict()
        for name, counter in counters:
            df = counter2df(counter)
            dfs[name] = df

            if path is not None:
                f = os.path.join(path, "label_summary_{}.csv".format(name))
                df.to_csv(f)

        return dfs
