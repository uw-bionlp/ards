



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

# Define experiment and load ingredients
ex = Experiment('step005_text_import')


'''

scp -r /data/users/lybarger/radiology_ie/analyses/step010_corpora_unlabeled/pna_cpis_FAST_FAILED/brat/  lybarger@web79.iths.org:/data/www/brat/data/chestxrayPNACPIS/

scp -r /data/users/lybarger/radiology_ie/analyses/step010_corpora_unlabeled/pna_cpis/brat_new_anno/  lybarger@web79.iths.org:/data/www/brat/data/ARDS/

scp -r /data/users/lybarger/radiology_ie/analyses/step010_corpora_unlabeled/COVID_XRAY/brat/  lybarger@web79.iths.org:/data/www/brat/data/ARDS/


chmod -f -R 777 /data/www/brat/data/ARDS/

'''

def load_exclude(fn):

    with open(fn, 'r') as f:
        X = json.load(f)

    X = [tuple(x) for x in X]

    return X

@ex.config
def cfg():

    # project name


    project = constants.PULMONARY

    #source = constants_pulmonary.COVID_XRAY
    source = constants_pulmonary.EMERGE_XRAY


    dir = paths_pulmonary.text_import


    # description
    description = source

    fast_run = False


    if fast_run:
        description += '_FAST'

    # Destination folder
    destination = os.path.join(dir, description)

    # Minimum number of sentences for tagging
    num_examples = 10

    sampling = None
    exclude = []


    source_params = None
    annotators = None


    id2filename = None

    if source == constants_pulmonary.COVID_XRAY:
        source_params = {}
        source_params['source'] = paths_pulmonary.covid_xray_notes
        source_params['linebreak_bound'] = True

        exclude = [ \
            (35, 7, 1),  (36, 3, 1),  (108, 2, 1), (115, 5, 1),
            (118, 2, 1), (123, 2, 1), (138, 6, 1), (139, 0, 1),
            (184, 4, 1), (203, 1, 1)]

        footer = constants_pulmonary.FOOTER
        sampling = []

        sampling.append({'round': 1, 'size': 20,  'seed': 1, "anno_type": 'multiple', "footer":footer, 'kwargs':{'day_range': (0,7), 'rep_range':None}})
        sampling.append({'round': 2, 'size': 200,  'seed': 2, "anno_type": 'single', "footer":footer, 'kwargs':{'day_range': (0,7), 'rep_range':None}})
        #sampling.append({'round': 3, 'size': 100,  'seed': 2, "anno_type": 'single', "footer":footer, 'kwargs':{'day_range': (0,7), 'rep_range':None}})
        annotators = ["Pavan", "Mark", "Linzee", "Matthew"]

        id2filename = corpus_xray.id2filename


    elif source == constants_pulmonary.EMERGE_XRAY:
        source_params = {}
        source_params['source'] = paths_pulmonary.emerge_xray_notes
        source_params['linebreak_bound'] = True

        exclude_file = '/home/lybarger/clinical_extractors/analyses_pulmonary/step007_compare_docs/covid_vs_emerge/exclude_emerge_xray.json'

        exclude = load_exclude(exclude_file)

        footer = constants_pulmonary.FOOTER
        sampling = []
        sampling.append({'round': 3, 'size': 300,  'seed': 1, "anno_type": 'single', "footer":footer, 'kwargs':{}})
        #sampling.append({'round': 4, 'size': 200,  'seed': 2, "anno_type": 'single', "footer":footer, 'kwargs':{}})
        #sampling.append({'round': 5, 'size': 200,  'seed': 3, "anno_type": 'single', "footer":footer, 'kwargs':{}})

        annotators = ["Linzee", "Matthew"]

        id2filename = corpus_emerge_xray.id2filename

    else:
        raise ValueError(f"invalid source: {source}")



    # Output file
    corpus_fn = os.path.join(destination, constants.CORPUS_FILE)

    setup_dir(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



def sample_check(path_samples, path_out, ext='txt'):
    files = Path(path_samples).glob('**/*.{}'.format(ext))
    files = [file.relative_to(path_samples) for file in files]

    counter = Counter()
    for file  in files:
        dir = file.parent.parent
        id = file.stem
        counter[id] += 1

    df = pd.DataFrame(counter.items(), columns=["id", "counts"])
    df = df.sort_values("counts", ascending=False)
    f = os.path.join(path_out, "sample_check_by_id.csv")
    df.to_csv(f, index=False)

    counter = Counter([v for k, v in counter.items()])
    df = pd.DataFrame(counter.items(), columns=["occurrences", "counts"])
    df = df.sort_values("counts", ascending=False)
    f = os.path.join(path_out, "sample_check_overall.csv")
    df.to_csv(f, index=False)

    return df

@ex.automain
def main(source, destination, corpus_fn, num_examples, source_params,
    sampling, exclude, annotators, id2filename):

    # Load and create corpus
    logging.info("Corpus loading...")
    if source == constants_pulmonary.COVID_XRAY:
        corpus = xray.load_corpus(**source_params)

        logging.info('-'*72)
        logging.info('Only day 0')
        logging.info('-'*72)
        corpus.summary(path=destination, day_range=(0,0), rep_range=None)

        logging.info('-'*72)
        logging.info('Only day in [0-7]')
        logging.info('-'*72)
        corpus.summary(path=destination, day_range=(0,7), rep_range=None)

        logging.info('-'*72)
        logging.info('All notes')
        logging.info('-'*72)
        corpus.summary(path=destination, day_range=None,  rep_range=None)

    elif source == constants_pulmonary.EMERGE_XRAY:
        corpus = xray.load_emerge_corpus(**source_params)

    else:
        raise ValueError("Incorrect corpus:\t{}".format(corpus))
    logging.info("Corpus loaded")


    # Save examples for review
    example_dir = os.path.join(destination, "Examples")
    make_and_clear(example_dir, recursive=True)
    corpus.write_examples(example_dir, num_examples=num_examples)

    corpus.summary(path=destination)
    corpus.write_ids(destination)



    exclude =  [tuple(e) for e in exclude]

    if sampling is not None:
        logging.info('Sampling')

        # Sampling
        brat_dir = os.path.join(destination, "brat")
        make_and_clear(brat_dir, recursive=True)

        sampled = []
        for params in sampling:
            dir_ = os.path.join(brat_dir, 'round{:0>2d}'.format(params['round']))
            os.mkdir(dir_)

            logging.info('')
            logging.info('Round:\t{}'.format(params['round']))

            docs = corpus.random_sample( \
                    size = params['size'],
                    exclude = exclude,
                    seed = params['seed'],
                    path = dir_,
                    brat = True,
                    footer = params['footer'],
                    annotators = annotators,
                    anno_type = params['anno_type'],
                    **params['kwargs'])

            exclude.extend(list(docs.keys()))

            for id in docs:
                if id2filename is not None:
                    id = id2filename(id)
                sampled.append((params['round'], id))

        sample_check(brat_dir, destination)

        fn = os.path.join(destination, 'sampled_documents.csv')
        df = pd.DataFrame(sampled, columns=['round', 'id'])
        df.to_csv(fn)

    # Save corpus
    logging.info("Saving to disk...")
    joblib.dump(corpus, corpus_fn)
    logging.info("Saving complete")

    return True
