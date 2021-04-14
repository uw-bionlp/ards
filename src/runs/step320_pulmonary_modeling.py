
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

from config.constants import CV, FIT, PREDICT, SCORE, PROB, RELATIONS, SENT_LABELS, DOC_LABELS

# Define experiment and load ingredients
ex = Experiment('step320_pulmonary_modeling')

'''

python3 runs/step320_pulmonary_modeling.py with description='baseline' mode='prob' binary_doc_map=True model_dir='/home/lybarger/clinical_extractors/analyses_pulmonary/step320_pulmonary_modeling/fit/sent1+concat0+run0+bd1/'

python3 runs/step320_pulmonary_modeling.py with description='baseline' mode='predict' model_dir='/home/lybarger/clinical_extractors/analyses_pulmonary/step320_pulmonary_modeling/fit/sent1+concat0+run1/'


'''


@ex.config
def cfg():


    description = 'baseline'



    #description = 'baseline+crf'

    source = constants_pulmonary.COVID_XRAY
    dir = None


    #mode = CV
    #mode = FIT
    #mode = PREDICT
    #mode = SCORE
    mode = PROB
    model_dir='/home/lybarger/clinical_extractors/analyses_pulmonary/step320_pulmonary_modeling/fit/sent1+concat0+run0+bd1/'

    fast_run = False
    fast_count = 20 if fast_run else None

    source_dir = None

    n_splits = 3


    binary_doc_map = False




    if binary_doc_map:
        doc_map = constants_pulmonary.DOC_MAP_BINARY
        doc_definition = DOC_DEFINITION_BINARY
    else:
        doc_map = constants_pulmonary.DOC_MAP
        doc_definition = DOC_DEFINITION


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
    use_rnn = True
    use_doc_classifier = True
    use_span_classifier = False
    use_doc_features = False
    use_sent_objective = True
    concat_sent_scores = True


    projection_dim = 100

    if concat_sent_scores:
        assert use_sent_objective

    span_embed_dim = 50

    dropout_rnn = 0.0
    dropout_sent_classifier = 0.0
    dropout_doc_classifier = 0.0

    batch_size = 30

    linebreak_bound = True
    max_sent_count = 35
    keep_ws = True

    num_workers = 0
    xfmr_dim = 768
    rnn_size = 100
    h_size = rnn_size*2 if use_rnn else xfmr_dim
    loss_reduction = "sum"

    lr = 0.002

    hyperparams = {}
    hyperparams['use_rnn'] = use_rnn
    hyperparams['use_doc_classifier'] = use_doc_classifier
    hyperparams['use_span_classifier'] = use_span_classifier
    hyperparams['use_doc_features'] = use_doc_features


    rnn = {}
    rnn['input_size'] = xfmr_dim
    rnn['output_size'] = rnn_size
    rnn['type_'] = 'lstm'
    rnn['num_layers'] = 1
    rnn['dropout_output'] = dropout_rnn
    rnn['layer_norm'] = True
    hyperparams['rnn'] = rnn


    span_class_weights = OrderedDict()
    w = 100.0
    span_class_weights[REGION] =    [1.0, w, w]  # [NONE, PARENCHYMAL, EXTRAPARENCHYMAL]
    span_class_weights[SIDE] =      [1.0, w, w] # [NONE, UNILATERAL, BILATERAL]
    span_class_weights[SIZE] =      [1.0, w, w, w] # [NONE, SMALL, MODERATE, LARGE]
    span_class_weights[NEGATION] =  [1.0, w] # [NONE, NEGATION]

    relation_extractor = {}
    relation_extractor["entity_definition"] = entity_definition
    relation_extractor["input_dim"] = h_size
    relation_extractor["span_scorer_type"] = "span"
    relation_extractor["span_embed_project"] = True
    relation_extractor["span_embed_dim"] = span_embed_dim
    relation_extractor["span_embed_dropout"] = 0
    relation_extractor["span_scorer_hidden_dim"] = 50
    relation_extractor["span_scorer_dropout"] = 0
    relation_extractor["span_class_weights"] = None  # span_class_weights

    relation_extractor["spans_per_word"] = 2
    relation_extractor["relation_definition"] = relation_definition
    relation_extractor["role_hidden_dim"] = 50
    relation_extractor["role_output_dim"] = 2
    relation_extractor["role_dropout"] = 0
    relation_extractor["create_doc_vector"] = use_doc_features
    relation_extractor["doc_attention_dropout"] = 0
    relation_extractor["loss_reduction"] = loss_reduction
    hyperparams["relation_extractor"] = relation_extractor


    doc_classifier = {}
    doc_classifier["doc_definition"] = doc_definition
    doc_classifier["input_dim"] = h_size
    doc_classifier["query_dim"] = 100
    doc_classifier["use_ffnn"] = True
    doc_classifier["dropout_sent_classifier"] = dropout_sent_classifier
    doc_classifier["dropout_doc_classifier"] = dropout_doc_classifier
    doc_classifier["activation"] = 'tanh'
    doc_classifier["loss_reduction"] = loss_reduction
    doc_classifier["use_sent_objective"] = use_sent_objective
    doc_classifier["concat_sent_scores"] = concat_sent_scores
    doc_classifier["sent_definition"] = sent_definition
    doc_classifier["projection_dim"] = projection_dim
    hyperparams['doc_classifier'] = doc_classifier



    hyperparams['grad_max_norm'] = 1.0
    hyperparams["loss_reduction"] = loss_reduction

    dataset_params = {}
    dataset_params["pretrained"] = "emilyalsentzer/Bio_ClinicalBERT"
    dataset_params["max_length"] = 30
    dataset_params["max_wp_length"] = 50
    dataset_params["max_sent_count"] = max_sent_count
    dataset_params["linebreak_bound"] = linebreak_bound
    dataset_params["keep"] = 'mean'
    dataset_params["max_span_width"] = 6
    dataset_params["document_definition"] = doc_definition
    dataset_params["sent_definition"] = sent_definition
    dataset_params["entity_definition"] = entity_definition
    dataset_params["relation_definition"] = relation_definition
    dataset_params["pad_start"] = True
    dataset_params["pad_end"] = True
    dataset_params["keep_ws"] = False

    dataloader_params = {}
    dataloader_params['batch_size'] = batch_size
    dataloader_params['num_workers'] = num_workers

    optimizer_params = {}
    optimizer_params['lr'] = lr


    tokenization_params = {}
    tokenization_params['max_length'] = dataset_params["max_length"]
    tokenization_params['max_sent_count'] = dataset_params["max_sent_count"]
    tokenization_params['linebreak_bound'] = linebreak_bound
    tokenization_params['pad_start'] = dataset_params["pad_start"]
    tokenization_params['pad_end'] = dataset_params["pad_end"]


    num_workers = 6
    num_epochs = 300

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



@ex.automain
def main(source, destination, \
        hyperparams,
        dataset_params,
        dataloader_params,
        optimizer_params,
        num_workers,
        num_epochs,
        device,
        doc_map,
        n_splits, mode, model_dir, fast_count, side_swap, tokenization_params, sent_definition,
        linebreak_bound, max_sent_count, keep_ws):

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

    #ids_test = corpus.ids(include = [constants.TEST])


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



    model_class = ModelXray

    if mode in [CV, FIT]:

        model = model_class( \
                hyperparams = hyperparams,
                dataset_params = dataset_params,
                dataloader_params = dataloader_params,
                optimizer_params = optimizer_params,
                num_workers = num_workers,
                num_epochs = num_epochs)

        if mode == CV:
            scores = model.fit_cv(X=X_train, y=y_train, device=device, path=destination, n_splits=n_splits)

        elif mode == FIT:

            model.fit(X=X_train, y=y_train, device=device, path=destination)
            y_pred, scores = model.score(X=X_test, y=y_test, device=device, path=destination)

            model.save(path=destination)


    elif mode in [SCORE, PREDICT, PROB]:

        model = load_pretrained(model_class, model_dir, param_map=None)

        if mode == SCORE:
            y_pred, scores = model.score(X=X_test, y=y_test, device=device, path=destination)

        elif mode == PREDICT:
            y_pred = model.predict(X=X_test, device=device, path=destination)

        elif mode == PROB:

            y_prob = model.prob(X=X_test, y=y_test, device=device, path=destination)


    else:
        raise ValueError(f"Invalid mode: {mode}")


    return 'Successful completion'
