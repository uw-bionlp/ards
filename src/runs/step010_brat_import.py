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


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from corpus.corpus_brat import CorpusBrat
from corpus.corpus_brat_xray import CorpusBratXray
from corpus.corpus_brat_deid import CorpusBratDeid
from corpus.corpus_brat_symptoms import CorpusBratSymptoms
from utils.custom_observer import CustomObserver
from utils.proj_setup import make_and_clear
import config.constants as constants
import config.constants_pulmonary as constants_pulmonary
import config.paths as paths
import config.paths_deid as paths_deid
import config.paths_pulmonary as paths_pulmonary
import config.paths_symptoms as paths_symptoms

from corpus.tokenization import get_tokenizer, map2ascii, normalize_linebreaks, has_windows_linebreaks

import corpus.corpus_brat_xray as corpus_brat_xray



# Define experiment and load ingredients
ex = Experiment('step010_brat_import')

ID_FIELD = "id"
SUBSET_FIELD = "subset"
SOURCE_FIELD = "source"

@ex.config
def cfg():



    # Annotation source
    #source = constants.SYMPTOMS
    #source = constants.SDOH
    source = constants_pulmonary.COVID_XRAY
    #source = constants.SDOH_DEID
    #source = constants.SDOH_PARTIAL

    fast_run = False
    fast_count = 50 if fast_run else None

    source_dir = None

    skip = None
    source_tags = None
    source_original = None
    write_brat = False
    write_text = False
    map_ids = False
    corpus_object = CorpusBrat
    update_lb = False
    id2tags = None
    rm_extra_lb = False
    snap_textbounds = False
    linebreak_bound = True

    if source == constants.SDOH:
        source_dir = paths_deid.sdoh_brat
        source_tags = paths_deid.sdoh_doc_tags
        source_original = paths_deid.sdoh_original
        dir = paths_deid.brat_import
        write_brat = True
        write_text = True
        update_lb = True

    elif source == constants.SDOH_DEID:
        source_dir = paths_deid.sdoh_brat_deid
        dir = paths_deid.brat_import
        corpus_object = CorpusBratDeid

    elif source == constants.SDOH_PARTIAL:
        source_dir = paths.sdoh_brat_partial
        source_tags = paths.sdoh_doc_tags
        source_original = paths.sdoh_original
        dir = paths.brat_import
        write_brat = True
        write_text = True
        fast_run = True

    elif source == constants.SYMPTOMS:
        source_dir = paths_symptoms.symptoms_brat
        source_tags = paths_symptoms.symptoms_doc_tags
        #source_original = paths_symptoms.symptoms_original
        dir = paths_symptoms.brat_import
        corpus_object = CorpusBratSymptoms
        write_brat = False
        write_text = False

    elif source == constants_pulmonary.COVID_XRAY:
        source_dir = paths_pulmonary.brat_xray
        source_tags = paths_pulmonary.pulmonary_doc_tags
        dir = paths_pulmonary.brat_import
        corpus_object = CorpusBratXray
        id2tags = corpus_brat_xray.id2tags
        rm_extra_lb = True
        snap_textbounds = True
        skip = []
    else:
        ValueError("invalid source: {}".format(source))




    '''
    Paths
    '''
    if fast_run:
        destination = os.path.join(dir,  source+'_FAST_RUN')
    else:
        destination = os.path.join(dir,  source)

    # Destination file for corpus


    # Scratch directory
    make_and_clear(destination)

    # Create observers
    file_observ = FileStorageObserver.create(destination)
    cust_observ = CustomObserver(destination)
    ex.observers.append(file_observ)
    ex.observers.append(cust_observ)


def update_text(corpus, original_source, tokenizer, update_lb):

    original = json.load(open(original_source, "r"))


    # iterate over documents in corpus

    normalized_linebreaks = []
    pbar = tqdm(total=len(corpus))
    for i, (id, doc) in enumerate(corpus.docs_.items()):
        text = original[id]

        if update_lb and has_windows_linebreaks(text):
            normalized_linebreaks.append(id)
            text = normalize_linebreaks(text)

        text = map2ascii(text)

        doc.update_text_whitespace(text, tokenizer)
        pbar.update(1)
    pbar.close()

    logging.info("Update text")
    logging.info("\tCount with Windows line breaks: {}".format(len(normalized_linebreaks)))
    logging.info("\tDocs with Windows line breaks:\n{}".format('\n'.join(normalized_linebreaks)))


def get_tags_map(path, \
            id_field=ID_FIELD, subset_field=SUBSET_FIELD,
            source_field=SOURCE_FIELD):

    df = pd.read_csv(path)

    tag_map = OrderedDict()
    id_map = OrderedDict()

    digits = str(len(str(len(df))))
    pat = '{:0>' + digits + 'd}'

    pbar = tqdm(total=len(df))

    for i, d in enumerate(df.to_dict(orient="records")):

        id = d[id_field]

        assert id not in tag_map
        tag_map[id] = []
        if subset_field in d:
            subset = d[subset_field]
            tag_map[id].append(subset)

        if source_field in d:
            source = d[source_field]
            tag_map[id].append(source)

        assert id not in id_map
        id_map[id] = '{}/{}/{}'.format(subset, source, pat.format(i + 1))

        pbar.update(1)
    pbar.close()
    return (id_map, tag_map)

def map_and_tag_corpus(corpus, source_tags, destination, fast_run, map_ids=False):

    # get tags  for subsets and source
    id_map, tag_map = get_tags_map(source_tags)

    fn = os.path.join(destination, 'id_map.json')
    with open(fn, 'w') as f:
        json.dump(id_map, f, indent=4)

    fn = os.path.join(destination, 'tag_map.json')
    with open(fn, 'w') as f:
        json.dump(tag_map, f, indent=4)

    if not fast_run:
        n = len(corpus)
        logging.warn(f"len(id_map) != n: {len(id_map)} vs {n}")
        #assert len(id_map) == n
        logging.warn(f"len(tag_map) != n: {len(tag_map)} vs {n}")
        #assert len(tag_map) == n


    ids = list(corpus.docs_.keys())
    pbar = tqdm(total=len(ids))
    for id in ids:

        doc = corpus[id]
        doc.tags = set(tag_map[id])

        if map_ids:

            # update idea in tags
            doc.id = id_map[id]
            # add doc under GNU id and remove old doc id
            corpus[id_map[id]] = doc
            del corpus[id]

        pbar.update(1)
    pbar.close()

    # make sure corpus length is unchanged
    if not fast_run:
        assert len(corpus) == n


def text_to_disk(corpus, destination, sub_dir="text"):
    '''
    Save corpus to disc as txt files
    '''
    dir = os.path.join(destination, sub_dir)
    make_and_clear(dir, recursive=True)
    for doc in corpus.docs():
        doc.write_text(dir)


def get_label_freq(events, evt_typ, arg_typ):

    n_docs = len(events)

    counts = Counter()
    for doc in events:
        for sent in doc:
            for evt in sent:
                if evt.type_ == evt_typ:
                    for arg in evt.arguments:
                        if arg.type_ == arg_typ:
                            if arg.label is None:
                                logging.warn("Label is None: {}".format(evt))
                            else:
                                counts[arg.label] += 1

    for k, v in counts.items():
        counts[k] = v/float(n_docs)

    return counts


@ex.automain
def main(source, destination, source_dir, source_tags, source_original, \
        fast_run, write_brat, write_text, corpus_object, fast_count,
            update_lb, id2tags, rm_extra_lb, snap_textbounds, linebreak_bound, skip, map_ids):

    # Create corpus and tokenizer
    logging.info('Corpus instantiate')
    corpus = corpus_object()
    tokenizer = get_tokenizer(linebreak_bound=linebreak_bound)

    logging.info('Importing data from:\t{}'.format(source_dir))
    corpus.import_dir(source_dir, \
                    tokenizer = tokenizer,
                    n = fast_count,
                    rm_extra_lb = rm_extra_lb,
                    skip = skip)



    # Map and tag corpus
    if source_original is not None:
        logging.info('Updating text:\t{}'.format(source_original))
        update_text(corpus, source_original, tokenizer, update_lb)

    # Apply id and tag map
    if source_tags is not None:
        logging.info('Mapping and tagging corpus:\t{}'.format(source_tags))
        map_and_tag_corpus(corpus, source_tags, destination, fast_run, map_ids=map_ids)

    # Write corpus to disk in BRAT format
    if write_brat:
        logging.info('Writing BRAT')
        dir = os.path.join(destination, 'brat')
        corpus.write_brat(dir)

    if write_text:
        logging.info('Writing text')
        text_to_disk(corpus, destination)

    if id2tags is not None:
        corpus.ids2tags(id2tags, exclude=None)

    if snap_textbounds:
        corpus.snap_textbounds(exclude=None)

    corpus.histogram(path=destination)
    corpus.quality_check(path=destination)
    corpus.annotation_summary(path=destination)
    corpus.label_summary(path=destination)
    corpus.tag_summary(path=destination, exclude=None)

    dir = os.path.join(destination, 'tokenization_examples')
    corpus.tokenization_examples(path=dir, n=20)


    # Save annotated corpus
    logging.info('Saving corpus')
    fn_corpus = os.path.join(destination, constants.CORPUS_FILE)
    joblib.dump(corpus, fn_corpus)


    return 'Successful completion'
