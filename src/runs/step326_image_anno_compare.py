
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
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.custom_observer import CustomObserver

from utils.proj_setup import make_and_clear
import config.constants as constants
from config.constants import PREDICTIONS_FILE, TRAIN, TEST
import config.constants_pulmonary as constants_pulmonary
import config.paths as paths
import config.paths_pulmonary as paths_pulmonary
from data_loaders.xray import load_xray_images
from layers.pretrained import load_pretrained
from models.model_xray import ModelXray
from utils.misc import nest_list
from config.constants_pulmonary import NONE, PRESENT, UNILATERAL, BILATERAL
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL
from config.constants_pulmonary import NONE, PRESENT, UNILATERAL, BILATERAL
from data_loaders.xray import INFILTRATES_IMAGE
from corpus.corpus_xray import parse_filename

# Define experiment and load ingredients
ex = Experiment('step326_image_anno_compare')


@ex.config
def cfg():



    description = None


    use_binary = False
    description = f'binary_{int(use_binary)}'


    source = constants_pulmonary.XRAY_IMAGES

    source_corpus_text = '/home/lybarger/clinical_extractors/analyses_pulmonary/step005_text_import/covid_xray/corpus.pkl'

    source_corpus_anno = '/home/lybarger/clinical_extractors/analyses_pulmonary/step010_brat_import/covid_xray/corpus.pkl'



    if use_binary:
        source_model = '/home/lybarger/clinical_extractors/analyses_pulmonary/step320_pulmonary_modeling/fit/sent1+concat0+run0+bd1/'
        doc_map = constants_pulmonary.DOC_MAP_BINARY
    else:
        source_model = '/home/lybarger/clinical_extractors/analyses_pulmonary/step320_pulmonary_modeling/fit/sent1+concat0+run0/'
        doc_map = constants_pulmonary.DOC_MAP



    source_image = paths_pulmonary.xray_quadrant_interp



    load_predictions = False



    #file = constants.CORPUS_FILE
    #source = os.path.join(source_dir, source, file)

    device = 0



    '''
    Paths
    '''
    destination = os.path.join(paths_pulmonary.image_anno_comp, description)

    # Scratch directory
    make_and_clear(destination)


    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)





def get_image_anno(source, doc_map=None):

    logging.info(f"="*72)
    logging.info(f"Image annotations")
    logging.info(f"="*72)

    df = load_xray_images(source)

    image_study_ids = df['study_id'].to_list()
    image_accessions = df['accession'].to_list()
    assert len(image_accessions) == len(set(image_accessions))

    logging.info(f"Study_id count: {len(image_study_ids)}")
    logging.info(f"Accession count: {len(image_accessions)}")

    image_ids = list(zip(image_study_ids, image_accessions))


    if doc_map is not None:
        df['infiltrates_image'] = df['infiltrates_image'].map(doc_map)

    return (df, image_ids)



def get_text(source):


    '''
    Text
    '''

    logging.info(f"="*72)
    logging.info(f"Text")
    logging.info(f"="*72)

    corpus = joblib.load(source)

    text_dict = OrderedDict()
    map = OrderedDict()

    for doc in corpus.docs():
        study_id, day, rep = doc.id
        accession = doc.accession

        k = (study_id, accession)

        text_dict[k] = doc.text()
        map[(study_id, day, rep)] = k

    logging.info(f"Document count, all: {len(text_dict)}")

    return (text_dict, map)


def get_gold_anno(source, target_ids, map, doc_map, exclude='round03'):

    '''
    Gold annotations
    '''

    logging.info(f"="*72)
    logging.info(f"Gold annotations")
    logging.info(f"="*72)

    corpus = joblib.load(source)

    train_docs = corpus.docs(include=TRAIN, exclude=exclude)
    test_docs =  corpus.docs(include=TEST, exclude=exclude)
    all_docs = train_docs + test_docs

    train_ids = [map[parse_filename(doc.id)] for doc in train_docs]
    test_ids =  [map[parse_filename(doc.id)] for doc in test_docs]
    all_ids = train_ids + test_ids
    assert len(set(all_ids)) == len(all_ids)

    logging.info(f"Gold count, in train: {len(train_ids)}")
    logging.info(f"Gold count, in test:  {len(test_ids)}")
    logging.info(f"Gold count, in all:   {len(all_ids)}")



    y_true = OrderedDict()
    for doc in all_docs:
        study_id, day, rep = parse_filename(doc.id)

        k = map[(study_id, day, rep)]
        y_true[k] = doc.y(doc_map=doc_map)[constants.DOC_LABELS]
        doc_keys = y_true[k].keys()

    labels = []
    in_train = []
    in_test = []
    for study_id, accession in target_ids:
        k = (study_id, accession)

        if k in y_true:
            d = y_true[k]

        else:
            d = OrderedDict([(k, None) for k in doc_keys])
        labels.append(d)

        in_train.append(int(k in train_ids))
        in_test.append(int(k in test_ids))

    n_train = sum(in_train)
    n_test = sum(in_test)
    n_all = n_train + n_test

    logging.info(f"Gold in target count, in train:   {n_train}")
    logging.info(f"Gold in target count, in test:    {n_test}")
    logging.info(f"Gold in target count, in all:     {n_all}")


    labels = nest_list(labels)

    return (labels, in_train, in_test)


def get_predictions(source_model, device, text_dict, target_ids, path, load_predictions=False):

    logging.info(f"="*72)
    logging.info(f"Predictions")
    logging.info(f"="*72)



    logging.info(f"Document count, all: {len(text_dict)}")


    text = []
    for study_id, accession in target_ids:

        k = (study_id, accession)

        text.append(text_dict[k])

    logging.info(f"Document count, target: {len(text)}")


    dir = os.path.join(path, 'predictions')
    f = os.path.join(dir, PREDICTIONS_FILE)

    if load_predictions:
        y = joblib.load(f)
    else:
        model = load_pretrained(ModelXray, source_model)
        y = model.predict(X=text, device=device)


        make_and_clear(dir)
        joblib.dump(y, f)


    labels = [y_[constants.DOC_LABELS] for y_ in y]
    labels = nest_list(labels)

    return labels


@ex.automain
def main(source, destination, source_model, source_image, source_corpus_text, \
        device, load_predictions, source_corpus_anno, doc_map, use_binary):


    # Image annotations
    df, image_ids = get_image_anno(source_image, doc_map=doc_map)

    # get text
    text_dict, map = get_text(source_corpus_text)

    # Gold annotations
    image_true, in_train, in_test = get_gold_anno( \
            source = source_corpus_anno,
            target_ids = image_ids,
            map = map,
            doc_map = doc_map,
            exclude = 'round03')

    # Predictions
    image_pred = get_predictions( \
                source_model = source_model,
                device = device,
                text_dict = text_dict,
                target_ids = image_ids,
                path = destination,
                load_predictions = load_predictions)



    n = len(df)
    assert n == len(in_train)
    df['in_train'] = in_train
    #df['not_train'] = (df['in_train'] == 0).astype(int)

    assert n == len(in_test)
    df['in_test'] = in_test

    df['in_any'] = (df['in_train'] | df['in_test']).astype(int)

    for k, v in image_true.items():
        assert n == len(v)
        df[f"{k}_gold"] = v

    for k, v in image_pred.items():
        assert n == len(v)
        df[f"{k}_pred"] = v



    f = os.path.join(destination, "labels.csv")
    df.to_csv(f)


    #combos_both = [('pred', INFILTRATES_IMAGE, 'infiltrates_pred'), ('gold', INFILTRATES_IMAGE, 'infiltrates_gold')]
    combos_gold = [('gold', INFILTRATES_IMAGE, 'infiltrates_gold')]
    combos_pred = [('pred', INFILTRATES_IMAGE, 'infiltrates_pred')]

    #        names         col         val
    runs = []
    runs.append(('not_train', 'in_train', 0, combos_pred))
    runs.append(('in_any', 'in_any', 1,     combos_gold))
    for run_name, col, val, combos in runs:

        df_temp = df[df[col] == val]


        for (combo_name, a, b) in combos:

            A = df_temp[a]
            B = df_temp[b]

            print(run_name)
            print(combo_name)
            print(A)
            print(B)

            ct = pd.crosstab(A, B)
            f = os.path.join(destination, f"crosstabs_{run_name}_{combo_name}.csv")
            ct.to_csv(f)



            report = classification_report(A, B, output_dict=True)
            df_report = pd.DataFrame(report).transpose()

            f = os.path.join(destination, f"scores_{run_name}_{combo_name}.csv")
            df_report.to_csv(f)









            A = [int(a == 'bilateral') for a in A]
            B = [int(b == 'bilateral') for b in B]
            print(A)
            print(B)



            #p, r, f1, _ = precision_recall_fscore_support(A, B, pos_label=BILATERAL, average='binary')
            p, r, f1, _ = precision_recall_fscore_support(A, B, pos_label=1, average='binary')
            df_prf = pd.DataFrame([(p, r, f1)], columns=['P', 'R', 'F1'])

            f = os.path.join(destination, f"prf_{run_name}_{combo_name}.csv")
            df_prf.to_csv(f)

            tn, fp, fn, tp = confusion_matrix(A, B).ravel()
            sensitivity = r
            specificity = tn / (tn+fp)
            x = [(tn,   fp,   fn,   tp,   sensitivity,   specificity)]
            l = ['TN', 'FP', 'FN', 'TP', 'sensitivity', 'specificity']
            df_n = pd.DataFrame(x, columns=l)

            f = os.path.join(destination, f"sensitivity_specificity_{run_name}_{combo_name}.csv")
            df_n.to_csv(f)


    return 'Successful completion'
