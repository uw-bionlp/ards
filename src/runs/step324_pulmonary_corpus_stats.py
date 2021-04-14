
from __future__ import division, print_function, unicode_literals

from sacred import Experiment
from sacred.observers import FileStorageObserver

import os
import re
import numpy as np
import json
import joblib
import pandas as pd
from collections import Counter, OrderedDict
import logging
from tqdm import tqdm
import copy
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear
import config.constants as constants
import config.constants_pulmonary as constants_pulmonary
import config.paths as paths
import config.paths_deid as paths_deid
import config.paths_pulmonary as paths_pulmonary
import corpus.corpus_brat_xray as corpus_brat_xray

from models.model_ards import ModelARDS
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL, LABEL_MAP, DOC_DEFINITION, ENTITY_DEFINITION, RELATION_DEFINITION
from config.constants_pulmonary import REGION, SIDE, SIZE, NEGATION, NONE, UNILATERAL, BILATERAL, PRESENT
from layers.pretrained import load_pretrained
from visualization.utils import font_adjust
from config.constants import CV, FIT, PREDICT, SCORE, ENTITIES, RELATIONS, DOC_LABELS, TYPE, SUBTYPE

# Define experiment and load ingredients
ex = Experiment('step324_pulmonary_corpus_stats')


@ex.config
def cfg():



    description = 'basic'

    source_dir = paths_pulmonary.brat_import
    source = constants_pulmonary.COVID_XRAY
    file = constants.CORPUS_FILE
    source = os.path.join(source_dir, source, file)

    labels = [INFILTRATES, EXTRAPARENCHYMAL]

    doc_map = constants_pulmonary.DOC_MAP


    doc_label_order = [INFILTRATES, EXTRAPARENCHYMAL]
    assertion_label_order = [NONE, PRESENT, UNILATERAL, BILATERAL]



    '''
    Paths
    '''
    destination = os.path.join(paths_pulmonary.stats, description)


    # Scratch directory
    make_and_clear(destination)


    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



#def doc_rank(doc_label_order, assertion_label_order):
#
#    i = 0
#    map = OrderedDict()
#    for doc_label in doc_label_order:
#        for assertion_label in assertion_label_order:
#            map[(doc_label, assertion_label)] = i
#            i += 1
#    return map


def doc_plot(counter, doc_label_order, assertion_label_order, path, \
    figsize = (2.3, 1.3),
    palette = 'tab10',
    alpha = 1.0,
    font_size = 8,
    dpi = 1200
    ):

    sns.set_theme(style="whitegrid")
    font_adjust(font_size=font_size)

    counts = [(type, subtype, count) for (type, subtype), count in counter.items()]
    df = pd.DataFrame(counts, columns=[TYPE, SUBTYPE, "count"])

    fig, axs = plt.subplots(figsize=figsize, nrows=1, ncols=2, sharey=True)

    for i, label in enumerate(doc_label_order):
        ax = axs[i]

        df_tmp = df[df[TYPE] == label]

        g = sns.barplot(x=df_tmp['subtype'], y=df_tmp['count'], order=assertion_label_order, ax=ax)
        g.set(ylabel=None)

        ax.set_xlabel(label, fontweight='bold')

        if i == 0:
            ax.set_ylabel('Count', fontweight='bold')

        ax.tick_params(axis='x', labelrotation=90)

    fn = os.path.join(path, 'document_label_histogram.png')
    fig.savefig(fn, dpi=dpi, bbox_inches='tight')


    fn = os.path.join(path, 'document_label_histogram.csv')
    df.to_csv(fn)

    return df





@ex.automain
def main(source, destination, doc_map, doc_label_order, assertion_label_order):

    logging.info("Source: {}".format(source))
    logging.info("Destition = {}".format(destination))

    corpus = joblib.load(source)




    labels = corpus.y(doc_map=doc_map)

    count_docs = 0
    count_doc_labels = Counter()
    count_entities = Counter()
    count_relations = Counter()
    for doc in labels:

        count_docs += 1

        for k, v in doc[DOC_LABELS].items():
            count_doc_labels[(k, v)] += 1


        for entity in doc[ENTITIES]:
            count_entities[entity.type_] += 1

        for relation in doc[RELATIONS]:
            count_relations[1] += 1

    doc_plot(count_doc_labels, doc_label_order, assertion_label_order, destination)

    logging.info(f"Cocument count: {count_docs}")

    logging.info("Entity count, average per note:")
    for k, v in count_entities.items():
        logging.info(f"{k} = {v/count_docs:.2f}")

    logging.info("Relation count, average per note:")
    for k, v in count_relations.items():
        logging.info(f"{k} = {v/count_docs:.2f}")






    return 'Successful completion'
