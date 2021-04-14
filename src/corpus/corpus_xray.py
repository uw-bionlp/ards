


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




from corpus.corpus import Corpus


def id2stem(study_id, day, rep):

    filename = 'study{:03d}_day{:03d}_rep{:01d}'.format(study_id, day, rep)

    return filename

def id2filename(id):
    return id2stem(*id)

def parse_filename(filename):

    stem = Path(filename).stem


    study_id, day, rep = tuple(stem.split('_'))

    study_id = int(re.sub('study', '', study_id))
    day      = int(re.sub('day', '', day))
    rep      = int(re.sub('rep', '', rep))

    return (study_id, day, rep)



def range2set(indices):

    if indices is None:
        return None
    else:
        return set(list(range(indices[0], indices[1] + 1)))


class CorpusXray(Corpus):
    '''
    Corpus container (collection of documents)
    '''
    def __init__(self):


        Corpus.__init__(self)




    #@override
    def doc_filter(self, day_range=None, rep_range=None, out_type='list', include=None, exclude=None):
        '''
        Get filtered set of documents
        '''

        # include is only included for compatibility but is not used here
        assert include is None

        # exclude is only included for compatibility but is not used here
        assert exclude is None

        day_set = range2set(day_range)
        rep_set = range2set(rep_range)

        # Loop on documents
        docs = OrderedDict()
        for (study_id, day, rep), doc in self.docs_.items():

            # Check day/rep
            if ((day_range is None) or (day in day_set)) and \
               ((rep_range is None) or (rep in rep_set)) :

                assert (study_id, day, rep) not in docs
                docs[(study_id, day, rep)] = doc

        return docs

    #@override
    def id2stem(self, id):
        '''
        Convert document ID to filename stem
        '''
        study_id, day, rep = id
        stem = id2stem(study_id, day, rep)

        return stem

    def accessions(self, **kwargs):
        '''

        '''
        accessions = []
        for doc in self.docs(out_type='list', **kwargs):
            accessions.append(doc.accession)
        return accessions


    #@override
    def counts(self, **kwargs):

        pc = self.patient_count(**kwargs)
        dc = self.doc_count(**kwargs)
        sc = self.sent_count(**kwargs)
        wc = self.word_count(**kwargs)
        columns = ['patient count', 'doc count', 'sent count', 'word count']
        df = pd.DataFrame([[pc, dc, sc, wc]], columns=columns)

        return df

    #@override
    def summary(self, path=None, **kwargs):
        '''
        Create corpus summary
        '''

        summary = []

        # Arguments
        summary.append('')
        summary.append('kwargs:\t')
        for k, v in kwargs.items():
            summary.append('\t{} = {}'.format(k, v))
        summary.append('')

        # Basic counts
        df_counts = self.counts(**kwargs)
        summary.append('')
        summary.append('Counts:\n{}'.format(df_counts))
        summary.append('')

        # Basic timing
        df_day_rep, df_day, df_rep = self.timing_dist(**kwargs)
        summary.append('')
        summary.append('Timing distribution:')
        summary.append('')
        summary.append('\tDays:\n{}'.format(df_day))
        summary.append('')
        summary.append('\tReps per day:\n{}'.format(df_rep))
        summary.append('')

        # Convert to string
        summary = '\n'.join(summary)
        logging.info(summary)


        if path is not None:

            fn = os.path.join(path, 'corpus_counts.csv')
            df_counts.to_csv(fn)

            fn = os.path.join(path, 'distribution_by_days_reps.csv')
            df_day_rep.to_csv(fn, index=False)

            fn = os.path.join(path, 'distribution_by_day.csv')
            df_day.to_csv(fn, index=False)

            fn = os.path.join(path, 'distribution_notes_per_day.csv')
            df_rep.to_csv(fn, index=False)

            fn = os.path.join(path, 'corpus_summary.txt')
            with open(fn,'w') as f:
                f.write(summary)

        return summary


    def patient_count(self, **kwargs):
        '''
        Get patient count
        '''
        docs = self.docs(out_type='dict', **kwargs)
        study_ids = set([study_id for study_id, day, rep in docs])
        return len(study_ids)


    def timing_dist(self, **kwargs):
        '''
        Get distribution of x-ray times/days
        '''

        study_ids, days, reps = zip(*self.docs(out_type='dict', **kwargs).keys())

        days_reps_counts = Counter(zip(days, reps))
        df_day_rep = pd.DataFrame.from_dict(days_reps_counts, orient='index').reset_index()
        df_day_rep = df_day_rep.rename(columns={'index':'day', 0:'count'})
        df_day_rep = df_day_rep.sort_values('day')
        df_day_rep['percentage'] = df_day_rep['count']/df_day_rep['count'].sum()

        day_counts = Counter(days)
        df_day = pd.DataFrame.from_dict(day_counts, orient='index').reset_index()
        df_day = df_day.rename(columns={'index':'day', 0:'count'})
        df_day = df_day.sort_values('day')
        df_day['percentage'] = df_day['count']/df_day['count'].sum()

        rep_counts = Counter(reps)
        df_rep = pd.DataFrame.from_dict(rep_counts, orient='index').reset_index()
        df_rep = df_rep.rename(columns={'index':'per day', 0:'count'})
        df_rep = df_rep.sort_values('per day')
        df_rep['percentage'] = df_rep['count']/df_rep['count'].sum()

        return (df_day_rep, df_day, df_rep)
