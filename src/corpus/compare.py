
import os
import difflib
from collections import OrderedDict, Counter
import logging
import numpy as np
from tqdm import tqdm


def length_sim(a, b):

    a = float(a)
    b = float(b)

    return min(b/a, a/b)



def compare_text(a, b, rm_ws=True, threshold=0.8):


    count_a = len(a)
    count_b = len(b)


    similarity = np.zeros((count_a, count_b)) - 1

    logging.info(f'a, count: {count_a}')
    logging.info(f'b, count: {count_b}')
    logging.info(f'similarity, shape: {similarity.shape}')



    if rm_ws:
        remove_ws = lambda docs: OrderedDict([(k, ''.join(t.split())) \
                                                    for k, t in docs.items()])
        a = remove_ws(a)
        b = remove_ws(b)


    Y = {}
    pbar = tqdm(total=count_a*count_b)
    for i, (key_a, text_a) in enumerate(a.items()):
        len_a  = len(text_a)
        for j, (key_b, text_b) in enumerate(b.items()):
            len_b  = len(text_b)

            #sim_quick = seq_matcher.real_quick_ratio()
            sim_quick = length_sim(len_a, len_b)

            if sim_quick >= threshold:
                seq_matcher = difflib.SequenceMatcher(None, text_a, text_b, autojunk=False)
                sim = seq_matcher.ratio()

            else:
                sim = sim_quick

            similarity[i, j] = sim

            pbar.update()

            Y[(key_a, key_b)] = sim
    pbar.close()

    return Y


def compare_corpora(corpus_a, corpus_b, len_threshold=0.8):




    docs_a = corpus_a.docs(out_type="dict")
    docs_b = corpus_b.docs(out_type="dict")


    get_text = lambda docs: OrderedDict([(k, doc.text()) for k, doc in docs.items()])

    text_a = get_text(docs_a)
    text_b = get_text(docs_b)

    similarity_dict = compare_text(text_a, text_b, threshold=len_threshold)


    return similarity_dict
