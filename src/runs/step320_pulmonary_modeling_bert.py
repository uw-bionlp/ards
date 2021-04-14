
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


from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear
import config.constants as constants
import config.constants_pulmonary as constants_pulmonary
import config.paths as paths
import config.paths_deid as paths_deid
import config.paths_pulmonary as paths_pulmonary
import corpus.corpus_brat_xray as corpus_brat_xray
from layers.text_class_utils import get_sent_labels_multi


from models.model_bert_tc import ModelBertTC
from models.dataset_bert_tc import tokenize_corpus
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL, LABEL_MAP, DOC_DEFINITION, SENT_DEFINITION
from config.constants_pulmonary import ENTITY_DEFINITION, RELATION_DEFINITION
from config.constants_pulmonary import ENTITY_DEFINITION_SWAP, RELATION_DEFINITION_SWAP
from config.constants_pulmonary import REGION, SIDE, SIZE, NEGATION
from layers.pretrained import load_pretrained

from config.constants import CV, FIT, PREDICT, SCORE, RELATIONS, SENT_LABELS

# Define experiment and load ingredients
ex = Experiment('step320_pulmonary_modeling')


@ex.config
def cfg():


    description = 'bert'



    #description = 'baseline+crf'

    source = constants_pulmonary.COVID_XRAY
    dir = None


    mode = CV
    #mode = FIT
    #mode = PREDICT
    #mode = SCORE

    model_dir = '/home/lybarger/clinical_extractors/analyses_pulmonary/step320_pulmonary_modeling/fit/baseline/'

    fast_run = False
    fast_count = 20 if fast_run else None

    source_dir = None

    n_splits = 3

    doc_map = constants_pulmonary.DOC_MAP


    if source == constants_pulmonary.COVID_XRAY:
        source_dir = paths_pulmonary.brat_import
        source = constants_pulmonary.COVID_XRAY
        source = os.path.join(source_dir, source, constants.CORPUS_FILE)
        output_dir = paths_pulmonary.modeling
    else:
        ValueError("invalid source: {}".format(source))


    side_swap = False
    if side_swap:
        entity_definition = ENTITY_DEFINITION_SWAP
        relation_definition = RELATION_DEFINITION_SWAP
    else:
        entity_definition = ENTITY_DEFINITION
        relation_definition = RELATION_DEFINITION

    sent_definition = SENT_DEFINITION


    '''
    Paths
    '''
    if fast_run:
        destination = os.path.join(output_dir, mode, description + '_FAST_RUN')
    else:
        destination = os.path.join(output_dir, mode, description)

    # Destination file for corpus

    # Scratch directory
    make_and_clear(destination)

    device = 0
    use_sent_objective = True
    concat_sent_scores = True
    span_embed_dim = 50
    batch_size = 4

    num_workers = 0


    max_sent_count = 35
    keep_ws = False
    linebreak_bound = True

    dropout_sent = 0.0
    dropout_doc = 0.0

    lr = 1e-5
    lr_ratio = 1.0

    pretrained = "emilyalsentzer/Bio_ClinicalBERT" # 'bert-base-uncased'
    doc_definition = DOC_DEFINITION
    sent_definition = SENT_DEFINITION
    grad_max_norm = 1.0
    loss_reduction = "sum"


    project_sent = True
    project_size = 200

    attention_query_dim = 100

    max_length = 60

    num_workers = 6
    num_epochs = 10

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



@ex.automain
def main(source, destination, \

        use_sent_objective,
        concat_sent_scores,
        num_workers,
        num_epochs,
        device,
        doc_map,
        n_splits, mode, model_dir, fast_count, side_swap,
        dropout_sent, dropout_doc,
        max_sent_count, keep_ws, linebreak_bound,
        doc_definition, sent_definition,
        pretrained, grad_max_norm, loss_reduction,
        batch_size, lr, attention_query_dim, max_length, lr_ratio,
        project_sent, project_size
        ):

    logging.info("Source: {}".format(source))
    logging.info("Destition = {}".format(destination))

    corpus = joblib.load(source)



    X_train, y_train = corpus.Xy( \
                        doc_map = doc_map,
                        include = [constants.TRAIN],
                        side_swap = side_swap)

    X_test,  y_test  = corpus.Xy( \
                        doc_map = doc_map,
                        include = [constants.TEST],
                        side_swap = side_swap)

    _, sent_offsets_train, _ = tokenize_corpus( \
                            documents = X_train,
                            max_sent_count = max_sent_count,
                            linebreak_bound = linebreak_bound,
                            keep_ws = keep_ws)

    _, sent_offsets_test, _ = tokenize_corpus( \
                            documents = X_test,
                            max_sent_count = max_sent_count,
                            linebreak_bound = linebreak_bound,
                            keep_ws = keep_ws)


    for i, y in enumerate(y_train):
        y[SENT_LABELS] =  get_sent_labels_multi( \
                                    relations = y[RELATIONS],
                                    sent_offsets = sent_offsets_train[i],
                                    sent_definition = sent_definition,
                                    out_type = 'dict')

    for i, y in enumerate(y_test):
        y[SENT_LABELS] =  get_sent_labels_multi( \
                                    relations = y[RELATIONS],
                                    sent_offsets = sent_offsets_test[i],
                                    sent_definition = sent_definition,
                                    out_type = 'dict')



    if fast_count:
        X_train = X_train[20:fast_count+20]
        y_train = y_train[20:fast_count+20]

    model_class = ModelBertTC

    if mode in [CV, FIT]:

        model = model_class( \
                doc_definition = doc_definition,
                sent_definition = sent_definition,
                pretrained = pretrained,
                num_workers = num_workers,
                num_epochs = num_epochs,
                dropout_sent = dropout_sent,
                dropout_doc = dropout_doc,
                use_sent_objective = use_sent_objective,
                concat_sent_scores = concat_sent_scores,
                grad_max_norm = grad_max_norm,
                loss_reduction = loss_reduction,
                batch_size = batch_size,
                lr = lr,
                lr_ratio = lr_ratio,
                attention_query_dim = attention_query_dim,
                max_length = max_length,
                max_sent_count = max_sent_count,
                linebreak_bound = linebreak_bound,
                keep_ws = keep_ws,
                project_sent = project_sent,
                project_size = project_size)

        if mode == CV:
            scores = model.fit_cv(X_train, y_train, device=device, path=destination, n_splits=n_splits)

        elif mode == FIT:

            model.fit(X_train, y_train, device=device, path=destination)
            y_pred, scores = model.score(X_test, y_test, device=device, path=destination)
            #y_pred, scores = model.score(X_train, y_train, device=device, path=destination)
            #print("CHEATING "*500)

            model.save(path=destination)


    elif mode in [SCORE, PREDICT]:

        model = load_pretrained(model_class, model_dir, param_map=None)

        if mode == SCORE:
            y_pred, scores = model.score(X_train, y_train, device=device, path=destination)

        elif mode == PREDICT:
            y_pred = model.predict(X_train, device=device, path=destination)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    #
    #model.score(X_train, y_train, device=device, path=destination)

    #model.predict(X_train, device=device)

    return 'Successful completion'
