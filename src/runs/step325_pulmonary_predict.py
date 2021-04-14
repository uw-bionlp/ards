
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


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from layers.text_class_utils import get_sent_labels_multi

from corpus.tokenization import tokenize_corpus


from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear
import config.constants as constants
import config.constants_pulmonary as constants_pulmonary
import config.paths as paths
import config.paths_deid as paths_deid
import config.paths_pulmonary as paths_pulmonary
import corpus.corpus_brat_xray as corpus_brat_xray

from models.model_xray import ModelXray
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL, LABEL_MAP, DOC_DEFINITION, SENT_DEFINITION, DOC_DEFINITION_BINARY
from config.constants_pulmonary import ENTITY_DEFINITION, RELATION_DEFINITION
from config.constants_pulmonary import ENTITY_DEFINITION_SWAP, RELATION_DEFINITION_SWAP
from config.constants_pulmonary import REGION, SIDE, SIZE, NEGATION
from layers.pretrained import load_pretrained

# Define experiment and load ingredients
ex = Experiment('step325_pulmonary_predict')


@ex.config
def cfg():


    description = 'covid_xray'

    source = '/home/lybarger/clinical_extractors/analyses_pulmonary/step005_text_import/covid_xray/corpus.pkl'
    include = None
    exclude = None
    as_stem = True

    model_dir = '/home/lybarger/clinical_extractors/analyses_pulmonary/step320_pulmonary_modeling/fit/sent1+concat0+run1/'


    fast_run = True

    output_dir = paths_pulmonary.predict
    if fast_run:
        destination = os.path.join(output_dir, description + '_FAST_RUN')
    else:
        destination = os.path.join(output_dir, description)

    # Scratch directory
    make_and_clear(destination)

    device = 0

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



@ex.automain
def main(source, destination, model_dir, fast_run, device, include, exclude, as_stem):

    logging.info("Source: {}".format(source))
    logging.info("Destition = {}".format(destination))


    corpus = joblib.load(source)


    X = corpus.X(include=include, exclude=exclude)
    ids = corpus.ids(include=include, exclude=exclude)
    accessions = corpus.accessions(include=include, exclude=exclude)

    logging.info("Examples:")
    for i, (x, id, accession) in enumerate(zip(X, ids, accessions)):
        logging.info("")
        logging.info("="*72)
        logging.info(f"id={id}")
        logging.info(f"accession={accession}")
        logging.info("="*72)
        logging.info(f"\n{x}")
        if i > 2:
            logging.info("="*72)
            logging.info("")
            break

    assert len(X) == len(ids)

    if fast_run:
        X = X[0:10]
        ids = ids[0:10]
        accessions = accessions[0:10]

    model_class = ModelXray
    model = load_pretrained(model_class, model_dir, param_map=None)
    y_pred = model.predict(X=X, device=device, path=destination)

    assert len(y_pred) == len(ids)

    out = []
    for id, accession, y in zip(ids, accessions, y_pred):

        o = OrderedDict()
        study_id, day, rep = id
        o["study_id"] = study_id
        o["day"] = day if day < 999 else 'disch'
        o["rep"] = rep
        o["accession"] = accession
        for k, v in y[constants.DOC_LABELS].items():
            o[k] = (v)
        out.append(o)
    df = pd.DataFrame(out)
    df = df.sort_values(["study_id", "day", "rep"])
    f = os.path.join(destination, "predictions.csv")
    df.to_csv(f, index=False)


    return 'Successful completion'
