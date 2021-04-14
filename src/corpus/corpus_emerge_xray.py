


import math
import json
import os
import joblib
import re
import shutil
import pandas as pd
from multiprocessing import Pool
import traceback
from tqdm import tqdm
import numpy as np
from collections import OrderedDict, Counter
import logging
from pathlib import Path
from datetime import datetime



from corpus.corpus import Corpus


DATE_FORMAT = "%Y-%m-%d-%H-%M-%S"

#def id2stem(study_id, date, date_format=DATE_FORMAT):
def id2stem(study_id, date):

    #date_str = date.strftime(date_format)

    filename = 'study{:04d}_{}'.format(study_id, date)

    return filename

def id2filename(id):
    return id2stem(*id)

#def parse_filename(filename, date_format=DATE_FORMAT):
def parse_filename(filename):

    stem = Path(filename).stem

    study_id, date = tuple(stem.split('_'))

    study_id = int(re.sub('study', '', study_id))
    #date = datetime.strptime(date, date_format)

    return (study_id, date)




class CorpusEmergeXray(Corpus):
    '''
    Corpus container (collection of documents)
    '''
    def __init__(self):


        Corpus.__init__(self)



    #@override
    def id2stem(self, id):
        '''
        Convert document ID to filename stem
        '''
        study_id, date = id
        stem = id2stem(study_id, date)

        return stem


    #@override
    def counts(self, **kwargs):

        pc = self.patient_count(**kwargs)
        dc = self.doc_count(**kwargs)
        sc = self.sent_count(**kwargs)
        wc = self.word_count(**kwargs)
        columns = ['patient count', 'doc count', 'sent count', 'word count']
        df = pd.DataFrame([[pc, dc, sc, wc]], columns=columns)

        return df

    def patient_count(self, **kwargs):
        '''
        Get patient count
        '''
        docs = self.docs(out_type='dict', **kwargs)
        study_ids = set([study_id for study_id, _ in docs])
        return len(study_ids)
