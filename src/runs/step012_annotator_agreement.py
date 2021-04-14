from __future__ import division, print_function, unicode_literals

from sacred import Experiment
from sacred.observers import FileStorageObserver
import sys
import os
import re
import numpy as np
import json
import joblib
import pandas as pd
from collections import Counter, OrderedDict
import logging
from tqdm import tqdm
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corpus.agreement import annotator_agreement, label_distribution
from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear
import config.constants as constants
import config.constants_pulmonary as constants_pulmonary
import config.paths as paths
import config.paths_pulmonary as paths_pulmonary

from scoring.scorer import Scorer
from scoring.scorer_xray import ScorerXray

from scoring.scoring_utils import PRF

# Define experiment and load ingredients
ex = Experiment('step012_annotator_agreement')


@ex.config
def cfg():



    # Annotation source
    #source = constants.SYMPTOMS
    #source = constants.SDOH
    source = constants_pulmonary.COVID_XRAY

    description = 'all'

    fast_run = False

    source_dir = None



    index_round = 0
    index_annotator = 1
    index_note = 2

    round = None
    annotators = None


    scorer = Scorer

    label_spec = {}

    if source == constants.SDOH:
        pass
    elif source == constants.SDOH_PARTIAL:
        pass

    elif source == constants.SYMPTOMS:
        pass

    elif source == constants_pulmonary.COVID_XRAY:

        source_dir = paths_pulmonary.brat_import
        dir = paths_pulmonary.agreement
        if description == 'round01':
            target_rounds_aggree = ["round01"]
        elif description == 'round04':
            target_rounds_aggree = ["round04"]
        elif description == 'all':
            target_rounds_aggree = ["round01", "round04"]
        target_rounds_dist = ["round02"]
        #annotator_pairs = [('Mark', 'Linzee'),
        #                   ('Mark', 'Matthew'),
        #                   ('Linzee', 'Matthew')]
        annotator_pairs = [('Linzee', 'Matthew')]
        scorer = ScorerXray
        doc_map = constants_pulmonary.DOC_MAP

        label_spec = {'doc_map': doc_map}


    else:
        ValueError("invalid source: {}".format(source))



    source_corpus = os.path.join(source_dir, source, constants.CORPUS_FILE)


    '''
    Paths55309
    '''

    destination = os.path.join(dir,  source, description)

    # Destination file for corpus


    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



def plot_doc_aggree(df, path, \
    height = 2.0,
    aspect = 2.0,
    palette = 'tab10',
    alpha = 1.0):

    print('='*100)

    print(df)


    #font_paths = mpl.font_manager.findSystemFonts()
    #font_objects = mpl.font_manager.createFontList(font_paths)
    #font_names = [f.name for f in font_objects]
    #print(font_names)


    sns.set_theme(style="whitegrid")
    #sns.set_style({'font.family': 'Times New Roman'})


    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="type", y="F1", hue="subtype",
        ci="sd", palette=palette, alpha=alpha, height=height, aspect=aspect)
    #g.despine(left=True)
    g.set_axis_labels("Document labels", "F1")
    g.legend.set_title("")

    f = os.path.join(path, 'document_agreement.png')
    g.savefig(f, dpi=1200)


@ex.automain
def main(source, destination, source_corpus, \
    index_round, index_annotator, index_note, target_rounds_aggree,
    target_rounds_dist, annotator_pairs, scorer, label_spec):

    # Create corpus and tokenizer
    logging.info('Corpus instantiate')

    corpus = joblib.load(source_corpus)



    dfs = annotator_agreement(corpus, index_round, index_annotator, index_note, \
                    target_rounds_aggree, annotator_pairs, scorer, path=None, **label_spec)


    for name, df in dfs.items():
        print(name)
        print(df)

        f = os.path.join(destination, f'scores_{name}.csv')
        df.to_csv(f)

        #if name in ['entities_summary', 'entities_partial_summary']:
        #    df = PRF(df.sum(axis=0)).to_frame().T
        #    f = os.path.join(destination, f'scores_{name}_SUMMARY.csv')
        #    df.to_csv(f)


        #df = pd.read_csv(f)
        #plot_doc_aggree(df, path=destination)



    #label_distribution(corpus, index_round, index_annotator, index_note, \
    #            target_rounds_dist, annotator_pairs, scorer, path=destination, **label_spec)


    #f = os.path.join(destination, 'scores_doc_labels_SUMMARY.csv')
    #df = pd.read_csv(f)
    #plot_doc_aggree(df, path=destination)


    return 'Successful completion'
