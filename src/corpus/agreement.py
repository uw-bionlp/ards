


import logging
import pandas as pd
import os
from collections import OrderedDict, Counter

from config.constants import ANNOTATOR_A, ANNOTATOR_B, TYPE, SUBTYPE, ARG_1, ARG_2, ROLE, TP, FN, FP, P, R, F1
from scoring.scoring_utils import PRF

def get_docs_by_annotator(docs, index_round, index_annotator, index_note, target_rounds, annotator_pairs):

    '''
    Organize annotations by annotator


    '''

    logging.info('Get documents by annotator')

    annotators = set([x for ab in annotator_pairs for x in ab])

    docs_by_annotator = OrderedDict()

    # Loop on documents
    for i, doc in enumerate(docs):

        # Get annotation round, annotator, and Doc ID
        id = doc.id.split(os.sep)

        round = id[index_round]
        annotator = id[index_annotator]
        note = id[index_note]

        if (round in target_rounds) and (annotator in annotators):

            if annotator not in docs_by_annotator:
                docs_by_annotator[annotator] = OrderedDict()

            # Include doc
            docs_by_annotator[annotator][note] = doc

    logging.info("\tAnnotator keys:")
    for k in docs_by_annotator:
        logging.info("\t\t{}".format(k))

    counts = []
    logging.info("\tDoc count by annotator:")
    for annotator, docs in docs_by_annotator.items():
        logging.info('\t\t{} = {}'.format(annotator, len(docs)))

    return docs_by_annotator


def summarize_df(df, gb, path=None, name=None):

    df = df.groupby(gb).sum().reset_index()
    df = PRF(df)

    return df

def compare_annotations(annotator_pairs, docs_by_annotator, scorer, path=None, summary_cols = [TYPE, SUBTYPE, ARG_1, ARG_2], **kwargs):

    scorer = scorer()

    # Loop on gold-annotator pairs
    dfs_all = OrderedDict()
    for annotator_a, annotator_b in annotator_pairs:

        # Get relevant documents
        docs_a = docs_by_annotator[annotator_a]
        docs_b = docs_by_annotator[annotator_b]

        assert len(docs_a) == len(docs_b)

        logging.info('-'*72)
        logging.info('Annotator pair:\t{} - {} ({} docs)'.format( \
                        annotator_a, annotator_b, len(docs_a)))
        logging.info('-'*72)


        ids = list(docs_a.keys())
        #labels_a = [docs_a[id].labels() for id in ids]
        #labels_b = [docs_b[id].labels() for id in ids]
        labels_a = [docs_a[id].y(**kwargs) for id in ids]
        labels_b = [docs_b[id].y(**kwargs) for id in ids]

        dfs = scorer.fit(labels_a, labels_b)

        for name, df in dfs.items():
            df[ANNOTATOR_A] = annotator_a
            df[ANNOTATOR_B] = annotator_b

            if name not in dfs_all:
                dfs_all[name] = []

            dfs_all[name].append(df)

        if path is not None:
            for name, df in dfs.items():
                fn = os.path.join(path, 'scores_{}_{}_{}.csv'.format(name, annotator_a, annotator_b))
                df.to_csv(fn)

    for name, df in dfs_all.items():

        df = pd.concat(df)
        dfs_all[name] = df

        if path is not None:
            fn = os.path.join(path, 'scores_{}_ALL.csv'.format(name))
            df.to_csv(fn)

        cols = [c for c in summary_cols if c in df]

        if len(cols) > 0:
            df_summary = summarize_df(df, cols)

            if path is not None:
                fn = os.path.join(path, 'scores_{}_SUMMARY.csv'.format(name))
                df_summary.to_csv(fn)

    return dfs_all


def label_dist(annotator_pairs, docs_by_annotator, scorer, path=None, summary_cols = [TYPE, SUBTYPE, ARG_1, ARG_2],
    rm_cols = [TP, FN, FP, P, R, F1], **kwargs):

    scorer = scorer()

    # Loop on gold-annotator pairs
    dfs_all = OrderedDict()
    for annotator_a, annotator_b in annotator_pairs:

        # Get relevant documents
        docs_a = docs_by_annotator[annotator_a]
        docs_b = docs_by_annotator[annotator_b]

        assert len(docs_a) == len(docs_b)

        logging.info('-'*72)
        logging.info('Annotator pair:\t{} - {} ({} docs)'.format( \
                        annotator_a, annotator_b, len(docs_a)))
        logging.info('-'*72)


        ids = list(docs_a.keys())
        #labels_a = [docs_a[id].labels() for id in ids]
        #labels_b = [docs_b[id].labels() for id in ids]
        labels_a = [docs_a[id].y(**kwargs) for id in ids]
        labels_b = [docs_b[id].y(**kwargs) for id in ids]

        dfs = scorer.fit(labels_a, labels_b)

        for name, df in dfs.items():
            for c in rm_cols:
                del df[c]

        for name, df in dfs.items():
            df[ANNOTATOR_A] = annotator_a
            df[ANNOTATOR_B] = annotator_b

            if name not in dfs_all:
                dfs_all[name] = []

            dfs_all[name].append(df)

        if path is not None:
            for name, df in dfs.items():
                fn = os.path.join(path, 'hist_{}_{}_{}.csv'.format(name, annotator_a, annotator_b))
                df.to_csv(fn)


def annotator_agreement(corpus, index_round, index_annotator, index_note, \
                                    target_rounds, annotator_pairs, scorer, path, **kwargs):


    docs_by_annotator = get_docs_by_annotator( \
            docs = corpus.docs(exclude=None),
            index_round=index_round,
            index_annotator=index_annotator,
            index_note=index_note,
            target_rounds=target_rounds,
            annotator_pairs=annotator_pairs)

    dfs = compare_annotations(annotator_pairs, docs_by_annotator, scorer, path=path, **kwargs)


    return dfs

def label_distribution(corpus, index_round, index_annotator, index_note, \
                                    target_rounds, annotator_pairs, scorer, path, **kwargs):

    docs_by_annotator = get_docs_by_annotator( \
            docs = corpus.docs(),
            index_round=index_round,
            index_annotator=index_annotator,
            index_note=index_note,
            target_rounds=target_rounds,
            annotator_pairs=annotator_pairs)

    for annotator, docs in docs_by_annotator.items():
        docs_by_annotator[annotator] = OrderedDict([(i, doc) for i, (id, doc) in enumerate(docs.items())])

    label_dist(annotator_pairs, docs_by_annotator, scorer, path=path, **kwargs)
