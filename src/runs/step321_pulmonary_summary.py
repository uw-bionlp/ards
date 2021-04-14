
from __future__ import division, print_function, unicode_literals

from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path
import os
import re
import numpy as np
import json
import joblib
import pandas as pd
from collections import Counter, OrderedDict
import logging
from tqdm import tqdm
from scipy.stats import ttest_ind
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from scoring.scoring_utils import PRF
from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear
import config.constants as constants
import config.constants_pulmonary as constants_pulmonary
import config.paths as paths
import config.paths_pulmonary as paths_pulmonary
import corpus.corpus_brat_xray as corpus_brat_xray


from config.constants import CV, FIT, PREDICT, SCORE, DOC_LABELS, TYPE, F1, SENT_LABELS, SUBTYPE, NT, NP, TP, P, R
from config.constants_pulmonary import NONE, PRESENT, UNILATERAL, BILATERAL

# Define experiment and load ingredients
ex = Experiment('step321_pulmonary_summary')


MODEL = 'model'
RUN = 'run'
CATEGORY = 'category'
MICRO = "micro"
RANK = 'rank'

@ex.config
def cfg():


    #mode = CV
    mode = FIT
    #mode = PREDICT
    #mode = SCORE


    #file_doc_scores = ["scores_doc_labels.csv", "scores_doc_labels_summary.csv", "scores_sent_labels_summary.csv"] #"scores_entities.csv", "scores_relations.csv"]
    file_doc_scores = "scores_doc_labels.csv" #"scores_entities.csv", "scores_relations.csv"]
    file_sent_scores = "scores_sent_labels_summary.csv"

    source_dirs = [os.path.join(paths_pulmonary.modeling, mode)]
    discrete_dir = '/home/lybarger/clinical_extractors/analyses_pulmonary/step322_pulmonary_discrete/ngrams/'

    if mode == FIT:
        source_dirs.append(discrete_dir)

    metric = F1
    destination = os.path.join(paths_pulmonary.summary, mode)


    suffix_pat = '\+run\d'

    # Destination file for corpus

    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



@ex.automain
def main(source_dirs, destination, file_doc_scores, file_sent_scores, suffix_pat, metric):



    # get all sub directories
    result_dirs = []
    for dir in source_dirs:
        result_dirs.extend([path for path in Path(dir).iterdir() if path.is_dir()])

    logging.info(f"Source directories:")
    for dir in source_dirs:
        logging.info(f"{dir}")
    logging.info(f"Directory count: {len(result_dirs)}")



    unnamed = 'Unnamed: 0'

    file_name = file_doc_scores
    dfs = []
    for dir in result_dirs:
        f = os.path.join(dir, file_name)
        name = str(dir.name)
        name_abbrev = re.sub(suffix_pat, '', name)

        if os.path.exists(f):

            df = pd.read_csv(f)

            if unnamed in df:
                del df[unnamed]
            df.insert(0, RUN, name)
            df.insert(0, MODEL, name_abbrev)

            df = df.fillna(0)
            dfg = PRF(df.groupby(TYPE).agg({MODEL:'first', RUN:'first', NT:'sum', NP:'sum', TP:'sum'}))
            dfg.reset_index(level=0, inplace=True)
            dfg.insert(0, SUBTYPE, MICRO)

            df = pd.concat([df, dfg])
            df[RANK] = df[SUBTYPE].apply(lambda row: [NONE, PRESENT, UNILATERAL, BILATERAL, MICRO].index(row))
            df = df.sort_values([MODEL, RUN, TYPE, RANK])

            print(df)

            dfs.append(df)


    if len(dfs) > 0:
        df = pd.concat(dfs)

        df.sort_values([MODEL, RUN, TYPE, RANK], inplace=True)

        f = os.path.join(destination, file_name)
        df.to_csv(f)

        f = os.path.join(destination, f'{file_name}_micro.csv')
        df_ = df[df[SUBTYPE] == MICRO]
        df_.to_csv(f)

        df[CATEGORY] = df.apply(lambda row: f'{row[TYPE]}-{row[SUBTYPE]}', axis = 1)

        summary = []
        for category in  df[CATEGORY].unique():
            df_temp = df[df[CATEGORY] == category]
            models = df_temp[MODEL].unique()
            stds = OrderedDict()
            pvals = OrderedDict()
            for a in models:

                A = df_temp[df_temp[MODEL] == a][metric]

                d = OrderedDict()
                d[CATEGORY] = category
                d[MODEL] = a
                d['mean'] = A.mean()
                d['std'] = A.std()

                for b in models:
                    B = df_temp[df_temp[MODEL] == b][metric]

                    _, pval = ttest_ind(A, B, equal_var=False)
                    d[f'Pval-{b}'] = pval

                summary.append(d)

        df = pd.DataFrame(summary)
        f = os.path.join(destination, f'summary.csv')
        df.to_csv(f)




    file_name = file_sent_scores
    dfs = []
    for dir in result_dirs:
        f = os.path.join(dir, file_name)
        name = str(dir.name)
        name_abbrev = re.sub(suffix_pat, '', name)

        if os.path.exists(f):
            df = pd.read_csv(f)
            if unnamed in df:
                del df[unnamed]
            df.insert(0, RUN, name)
            df.insert(0, MODEL, name_abbrev)
            dfs.append(df)

    if len(dfs) > 0:
        df = pd.concat(dfs)
        f = os.path.join(destination, file_name)
        df.to_csv(f)


    return 'Successful completion'
