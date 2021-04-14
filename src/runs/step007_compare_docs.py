



from sacred import Experiment
from sacred.observers import FileStorageObserver
import os
import joblib
import json
import pandas as pd
import shutil
import collections
import copy
import re
import logging
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.proj_setup import setup_dir
from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear
from data_loaders import xray
from config import paths, paths_pulmonary
from config import constants, constants_pulmonary
import corpus.corpus_xray as corpus_xray
import corpus.corpus_emerge_xray as corpus_emerge_xray
from corpus.compare import compare_corpora



# Define experiment and load ingredients
ex = Experiment('step007_compare_docs')



@ex.config
def cfg():

    description = 'covid_vs_emerge'

    source_a = '/home/lybarger/clinical_extractors/analyses_pulmonary/step005_text_import/covid_xray/corpus.pkl'
    source_b = '/home/lybarger/clinical_extractors/analyses_pulmonary/step005_text_import/emerge_xray/corpus.pkl'

    name_a = constants_pulmonary.COVID_XRAY
    name_b = constants_pulmonary.EMERGE_XRAY





    exclude_threshold = 0.90
    len_threshold =     0.89

    dir = paths_pulmonary.compare_docs

    fast_run = False


    if fast_run:
        description += '_FAST'

    # Destination folder
    destination = os.path.join(dir, description)


    setup_dir(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)




def save_exclude(exclude, name, path):

    fn = os.path.join(path, f'exclude_{name}.json')
    with open(fn, 'w') as f:
        json.dump(exclude, f, indent=4)

    #fn = os.path.join(path, f'exclude_{name}.pkl')
    #joblib.dump(exclude, f)

@ex.automain
def main(destination, source_a, source_b, name_a, name_b, len_threshold, exclude_threshold):


    corpus_a = joblib.load(source_a)

    corpus_b = joblib.load(source_b)

    similarity_dict = compare_corpora(corpus_a, corpus_b, len_threshold=len_threshold)

    fn = os.path.join(destination, 'similarity_scores.pkl')
    joblib.dump(similarity_dict, fn)



    similarities = []
    all_a = set([])
    all_b = set([])
    exclude_a = set([])
    exclude_b = set([])
    for (a, b), v in similarity_dict.items():
        #a = corpus_a.id2stem(a)
        #b = corpus_b.id2stem(b)
        similarities.append((a, b, v))

        all_a.add(a)
        all_b.add(b)

        if v >= exclude_threshold:
            exclude_a.add(a)
            exclude_b.add(b)

    exclude_a = sorted(list(exclude_a))
    exclude_b = sorted(list(exclude_b))

    logging.info(f"Count, similarity vals: {len(similarity_dict)}")
    logging.info(f"Count, all A:           {len(all_a)}")
    logging.info(f"Count, all B:           {len(all_b)}")
    logging.info(f"Exclude threshold:      {exclude_threshold}")
    logging.info(f"Count, exclude A:       {len(exclude_a)}")
    logging.info(f"Count, exclude B:       {len(exclude_b)}")


    save_exclude(exclude_a, name_a, destination)
    save_exclude(exclude_b, name_b, destination)


    fn = os.path.join(destination, 'similarity_scores.csv')
    df = pd.DataFrame(similarities, columns=['a', 'b', 'similarity'])
    df.sort_values('similarity', inplace=True, ascending=False)
    df.to_csv(fn)






    return True
