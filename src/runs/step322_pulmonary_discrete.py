
from __future__ import division, print_function, unicode_literals

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

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
from config.constants_pulmonary import REGION, SIDE, SIZE, NEGATION
from layers.pretrained import load_pretrained

from config.constants import DOC_LABELS_SUMMARY, SENT_LABELS_SUMMARY, TYPE, SUBTYPE, ARG_1, ARG_2, ROLE, SUBTYPE_A, SUBTYPE_B
from scoring.scoring_utils import PRF
from config.constants import CV, FIT, PREDICT, SCORE, ENTITIES, RELATIONS, DOC_LABELS

# Define experiment and load ingredients
ex = Experiment('step322_pulmonary_discrete')


@ex.config
def cfg():


    #prediction_type = 'oracle'
    run = 0
    prediction_type = 'ngrams'



    exclude_size = False
    exclude = None
    description = f"{prediction_type}+run{run}"



    fast_run = False

    n_splits = 3

    doc_map = constants_pulmonary.DOC_MAP


    source_dir = paths_pulmonary.brat_import
    source = constants_pulmonary.COVID_XRAY
    source = os.path.join(source_dir, source, constants.CORPUS_FILE)

    output_dir = paths_pulmonary.discrete


    labels = [INFILTRATES, EXTRAPARENCHYMAL]



    if prediction_type == 'oracle':

        model_type = 'random_forest'

        hyperparams = OrderedDict()
        hyperparams['n_estimators'] = 200
        hyperparams['max_depth'] = None
        hyperparams['min_samples_split'] = 2
        hyperparams['min_samples_leaf'] = 1
        hyperparams['min_weight_fraction_leaf'] = 0.0
        hyperparams['max_features'] = 'auto'
        hyperparams['max_leaf_nodes'] = None
        hyperparams['random_state'] = None
        hyperparams['ccp_alpha'] = 0.0
        hyperparams['n_jobs'] = 1
        #hyperparams['class_weight'] = {0:1, 1:1}

        tuned_parameters = OrderedDict()
        tuned_parameters['max_depth'] = [5, 10, 30] if fast_run else [2, 4, 6, 8, 10, 15, 20, 25, 30, 35]
        tuned_parameters['min_samples_split'] = [2, 4]  if fast_run else [2, 3, 4, 6, 8, 10, 12]
        tuned_parameters['n_estimators'] = [100] if fast_run else [50, 100, 200, 500]


        exclude_size = True
        if exclude_size:
            description = 'exclude_size'
            exclude = [SIZE]


    elif prediction_type == 'ngrams':


        model_type = 'svm'

        hyperparams = OrderedDict()
        hyperparams['C'] = 1.0
        #hyperparams['kernel'] = 'rbf'
        #hyperparams['degree'] = 3
        #hyperparams['gamma'] = 'scale'
        #hyperparams['coef0'] = 0.0
        #hyperparams['shrinking'] = True
        #hyperparams['probability'] = True
        #hyperparams['tol'] = 0.001
        #hyperparams['cache_size'] = 200
        #hyperparams['class_weight'] = None
        #hyperparams['verbose'] = False
        #hyperparams['max_iter'] = - 1,
        #hyperparams['decision_function_shape'] = 'ovr'
        #hyperparams['break_ties'] = False
        #hyperparams['random_state'] = None

        tuned_parameters = OrderedDict()
        tuned_parameters['C'] = [0.0001, 1.0, 1000.0] if fast_run else [0.0001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


    else:
        raise ValueError(f"invalid prediction type: {prediction_type}")

    '''
    Paths
    '''
    if fast_run:
        destination = os.path.join(output_dir, prediction_type, description + '_FAST_RUN')
    else:
        destination = os.path.join(output_dir, prediction_type, description)


    # Scratch directory
    make_and_clear(destination)


    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)


def get_X_as_relation_pairs(labels, exclude=None, columns=None):

    X = []
    for y in labels:
        relations = y[RELATIONS]
        d = {}
        for relation in relations:

            entity_a = relation.entity_a
            entity_b = relation.entity_b
            ta = entity_a.type_
            sa = entity_a.subtype
            tb = entity_b.type_
            sb = entity_b.subtype

            if (exclude is None) or ((ta not in exclude) and (tb not in exclude)):

                R = [ta, sa, tb, sb]
                R = ['null' if r is None else r for r in R]
                R = '-'.join(R)
                d[R] = 1

        X.append(d)

    df = pd.DataFrame(X)
    df = df.fillna(0)


    if columns is not None:
        for c in columns:
            if c not in df:
                df[c] = 0
        df = df[columns]

    return df


def get_X_ngrams(text, vectorizer=None):


    if vectorizer is None:
        vectorizer = TfidfVectorizer( \
                        stop_words = 'english',
                        use_idf = True)
        vectorizer.fit(text)

    X = vectorizer.transform(text).toarray()


    return (X, vectorizer)

def get_y(labels):
    y = [y[DOC_LABELS] for y in labels]
    df = pd.DataFrame(y)
    return df


@ex.automain
def main(source, destination, doc_map, n_splits, model_type, hyperparams, \
        tuned_parameters, labels, exclude, prediction_type):

    logging.info(f"Source: {source}")
    logging.info(f"Destination: {destination}")
    logging.info(f"Prediction type: {prediction_type}")

    corpus = joblib.load(source)

    labels_train = corpus.y(doc_map=doc_map, include=[constants.TRAIN])
    labels_test  = corpus.y(doc_map=doc_map, include=[constants.TEST])

    text_train = corpus.X(include=[constants.TRAIN])
    text_test  = corpus.X(include=[constants.TEST])

    if prediction_type == 'oracle':
        X_train = get_X_as_relation_pairs(labels_train, exclude=exclude, columns=None)
        X_test =  get_X_as_relation_pairs(labels_test,  exclude=exclude, columns=X_train.columns)
    elif prediction_type == 'ngrams':
        X_train, vectorizer = get_X_ngrams(text_train, vectorizer=None)
        X_test, vectorizer =  get_X_ngrams(text_test, vectorizer=vectorizer)

        logging.info(f"TF-IDF features, train: {X_train.shape}")
        logging.info(f"TF-IDF features, test: {X_test.shape}")
    else:
        raise ValueError(f"invalid prediction type: {prediction_type}")




    y_train = get_y(labels_train)
    y_test = get_y(labels_test)

    dfs_cv = []
    dfs_test = []
    for label in labels:
        logging.info("="*72)
        logging.info(f"Label: {label}")
        logging.info("="*72)


        model = ModelARDS( \
                        model_type = model_type,
                        hyperparams = hyperparams)


        best_params, df = model.fit_cv( \
                                        X = X_train,
                                        y = y_train[label],
                                        tuned_parameters = tuned_parameters,
                                        path = destination,
                                        n_splits = n_splits)

        df.insert(loc=0, column='type', value=[label]*len(df))
        dfs_cv.append(df)

        # update hyper parameterswith best parameters
        updated_params = copy.deepcopy(hyperparams)
        updated_params.update(best_params)

        model = ModelARDS( \
                        model_type = model_type,
                        hyperparams = updated_params)
        model.fit(X_train, y_train[label], path=destination)

        y_pred, df = model.score(X_test, y_test[label], path=destination)
        df.insert(loc=0, column='type', value=[label]*len(df))
        dfs_test.append(df)


    df = pd.concat(dfs_cv, axis=0)
    f = os.path.join(destination, "scores_doc_labels_cv_summary.csv")
    df.to_csv(f)


    df = pd.concat(dfs_test, axis=0)
    f = os.path.join(destination, "scores_doc_labels.csv")
    df.to_csv(f)

    print(df)

    df = PRF(df.groupby(TYPE).sum())
    f = os.path.join(destination, "scores_doc_labels_summary.csv")
    df.to_csv(f)


    return 'Successful completion'
