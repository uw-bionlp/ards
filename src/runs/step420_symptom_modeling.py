
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
import config.constants_symptoms as constants_symptoms
import config.paths as paths
import config.paths_symptoms as paths_symptoms
import corpus.corpus_brat_symptoms as corpus_brat_symptoms
from models.model_symptoms import ModelSymptoms


# Define experiment and load ingredients
ex = Experiment('step420_symptom_modeling')


@ex.config
def cfg():


    description = 'baseline'

    source = constants.SYMPTOMS

    dir = None

    fast_run = True

    source_dir = None

    include_train = [constants.TRAIN]


    if source == constants.SYMPTOMS:
        source_dir = paths_symptoms.brat_import
        if fast_run:
            source += '_FAST_RUN'
        source = os.path.join(source_dir, source, constants.CORPUS_FILE)
        output_dir = paths_symptoms.modeling
    else:
        ValueError("invalid source: {}".format(source))



    '''
    Paths
    '''
    if fast_run:
        destination = os.path.join(output_dir,  description + '_FAST_RUN')
    else:
        destination = os.path.join(output_dir,  description)

    # Destination file for corpus

    # Scratch directory
    make_and_clear(destination)

    device = 1
    use_rnn = True
    num_workers = 0
    xfmr_dim = 768
    lstm_size = 200
    h_size = lstm_size*2 if use_rnn else xfmr_dim
    loss_reduction = "sum"

    hyperparams = {}
    hyperparams['use_rnn'] = use_rnn

    rnn = {}
    rnn['input_size'] = xfmr_dim
    rnn['output_size'] = lstm_size
    rnn['type_'] = 'lstm'
    rnn['num_layers'] = 1
    rnn['dropout_output'] = 0.0
    hyperparams['rnn'] = rnn

    relation_extractor = {}
    relation_extractor["entity_definition"] = constants_symptoms.ENTITY_DEFINITION
    relation_extractor["input_dim"] = h_size
    relation_extractor["span_scorer_type"] = "span"
    relation_extractor["span_embed_project"] = True
    relation_extractor["span_embed_dim"] = 100
    relation_extractor["span_embed_dropout"] = 0.0
    relation_extractor["span_scorer_hidden_dim"] = 100
    relation_extractor["span_scorer_dropout"] = 0.0
    relation_extractor["span_class_weights"] = None

    relation_extractor["spans_per_word"] = 2
    relation_extractor["relation_definition"] = constants_symptoms.RELATION_DEFINITION
    relation_extractor["role_hidden_dim"] = 100
    relation_extractor["role_output_dim"] = 2
    relation_extractor["role_dropout"] = 0.0
    relation_extractor["loss_reduction"] = loss_reduction

    hyperparams["relation_extractor"] = relation_extractor
    hyperparams['grad_max_norm'] = 1.0
    hyperparams["loss_reduction"] = loss_reduction

    dataset_params = {}
    dataset_params["pretrained"] = "emilyalsentzer/Bio_ClinicalBERT"
    dataset_params["max_length"] = 30
    dataset_params["max_wp_length"] = 60
    dataset_params["linebreak_bound"] = True
    dataset_params["keep"] = 'mean'
    dataset_params["max_span_width"] = 6
    dataset_params["entity_definition"] = constants_symptoms.ENTITY_DEFINITION
    dataset_params["relation_definition"] = constants_symptoms.RELATION_DEFINITION

    dataloader_params = {}
    dataloader_params['batch_size'] = 100
    dataloader_params['num_workers'] = num_workers

    optimizer_params = {}
    optimizer_params['lr'] = 0.001

    num_workers = 6
    num_epochs = 100

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)



@ex.automain
def main(source, destination, include_train, \
        hyperparams,
        dataset_params,
        dataloader_params,
        optimizer_params,
        num_workers,
        num_epochs,
        device, fast_run):

    logging.info("Source: {}".format(source))
    logging.info("Destition = {}".format(destination))

    corpus = joblib.load(source)

    X_train, y_train = corpus.Xy(include=include_train)
    if fast_run:
        X_train = X_train[0:20]
        y_train = y_train[0:20]

    model = ModelSymptoms( \
            hyperparams = hyperparams,
            dataset_params = dataset_params,
            dataloader_params = dataloader_params,
            optimizer_params = optimizer_params,
            num_workers = num_workers,
            num_epochs = num_epochs)


    model.fit(X_train, y_train, device=device, path=destination)
    model.score(X_train, y_train, device=device, path=destination)

    #model.predict(X_train, device=device)

    return 'Successful completion'
