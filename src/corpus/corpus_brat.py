

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




from config.constants import ENCODING
from corpus.corpus import Corpus
from corpus.document_brat import DocumentBrat
from corpus.document_brat_xray import DocumentBratXray
from corpus.brat import get_brat_files
from corpus.tokenization import get_tokenizer, map2ascii, rm_extra_linebreaks, rm_footer
from utils.proj_setup import make_and_clear

def counter2df(counter):

    X = []

    for k, counts in counter.items():
        if isinstance(k, (list, tuple)):
            k = list(k)
        else:
            k = [k]
        X.append(k + [counts])

    df = pd.DataFrame(X)

    return df

class CorpusBrat(Corpus):

    def __init__(self, document_object=DocumentBrat):

        self.document_object = document_object

        Corpus.__init__(self)



    #def import_dir(self, path, tokenizer=None, doc_type=None):
    def import_dir(self, path, \
                        tokenizer = None,
                        n = None,
                        rm_extra_lb = False,
                        skip = None):

        '''
        Import BRAT directory
        '''

        # Find text and annotation files
        text_files, ann_files = get_brat_files(path)
        file_list = list(zip(text_files, ann_files))
        file_list.sort(key=lambda x: x[1])

        if n is not None:
            logging.warn("="*72)
            logging.warn("Only process processing first {} files".format(n))
            logging.warn("="*72)
            file_list = file_list[:n]


        if skip is not None:
            logging.warn("="*72)
            logging.warn(f"Skipping ids: {skip}")
            logging.warn("="*72)

        #file_list = [(f, a) for f, a in file_list if os.path.splitext(os.path.relpath(f, path))[0] == "04_uw_admit/Erica/uw_admit-40589"]

        pbar = tqdm(total=len(file_list))

        # Loop on annotated files
        for fn_txt, fn_ann in file_list:

            # Read text file
            with open(fn_txt, 'r', encoding=ENCODING) as f:
                text = f.read()

            text = map2ascii(text)

            if rm_extra_lb:
                text = rm_extra_linebreaks(text)

            # Read annotation file
            with open(fn_ann, 'r', encoding=ENCODING) as f:
                ann = f.read()

            # Use filename as ID
            id = os.path.splitext(os.path.relpath(fn_txt, path))[0]

            if (skip is None) or (id not in skip):

                doc = self.document_object( \
                    id = id,
                    text_ = text,
                    ann = ann,
                    tags = None,
                    patient = None,
                    date = None,
                    tokenizer = tokenizer,
                    )


                # Build corpus
                assert doc.id not in self.docs_
                self.docs_[doc.id] = doc

            pbar.update(1)

        pbar.close()


    def quality_check(self, path=None, **kwargs):

        X = []
        for doc in self.docs(out_type='list', **kwargs):
            X.extend(doc.quality_check())

        df = pd.DataFrame(X)

        if path is not None:
            f = os.path.join(path, "quality_check.csv")
            df.to_csv(f)

        return df



    def annotation_summary(self, path=None,**kwargs):

        counter = Counter()
        for doc in self.docs(out_type='list', **kwargs):
            counter += doc.annotation_summary()

        df = counter2df(counter)

        if path is not None:
            f = os.path.join(path, "annotation_summary.csv")
            df.to_csv(f)

        return df

    def label_summary(self, path=None, **kwargs):

        counter = Counter()
        for doc in self.docs(out_type='list', **kwargs):
            counter += doc.label_summary()

        df = counter2df(counter)

        if path is not None:
            f = os.path.join(path, "label_summary.csv")
            df.to_csv(f)

        return df

    def write_brat(self, path, **kwargs):

        make_and_clear(path, recursive=True)
        for doc in self.docs(**kwargs):
            doc.write_brat(path)

    def tag_summary(self, path, **kwargs):

        summary = []
        for doc in self.docs(**kwargs):
            summary.append((doc.id,doc.tags))

        df = pd.DataFrame(summary,columns=["id", "tags"])
        f = os.path.join(path, "tag_summary.csv")
        df.to_csv(f, index=False)

    def snap_textbounds(self, **kwargs):
        for doc in self.docs(**kwargs):
            doc.snap_textbounds()

    # OVERRIDE
    def y(self, **kwargs):

        y = []
        for doc in self.docs(**kwargs):
            y.append(doc.y())
        return y



    # OVERRIDE
    def Xy(self, **kwargs):

        X = []
        y = []
        for doc in self.docs(**kwargs):

            X_, y_ = doc.Xy()
            X.append(X_)
            y.append(y_)

        return (X, y)
