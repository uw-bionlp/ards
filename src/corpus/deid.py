

from collections import Counter, OrderedDict
from pathlib import Path
import json
import logging
import joblib
import os
import pandas as pd
import re
import difflib
from tqdm import tqdm
from copy import deepcopy

from corpus.labels import Entity
from config.constants_deid import DEID_PAT_LABEL, LABELS
from corpus.document_brat import remove_white_space_at_ends
from corpus.tokenization import simple_tokenization

def deid_summary(d, dir, description):

    counter_labels = Counter()
    counter_spans = Counter()
    counter_docs = Counter()
    for id, entities in d.items():
        counter_docs[id] = len(entities)
        for entity in entities:
            counter_labels[entity.subtype] += 1
            counter_spans[(entity.subtype, entity.text)] += 1

    dfs = {}
    dfs["docs"] = pd.DataFrame(counter_docs.items(), columns=["id", "count"])
    dfs["labels"] = pd.DataFrame(counter_labels.items(), columns=["subtype", "count"])
    dfs["spans"] = pd.DataFrame([(k, j, i) for (k, j), i  in counter_spans.items()], columns=["subtype", "span", "count"])

    for name, df in dfs.items():
        f = os.path.join(dir, "deid_summary_{}_{}.csv".format(description, name))
        df = df.sort_values("count", ascending=False)
        df.to_csv(f, index=False)

def load_automatic_deid(source, corpus, destination=None, extension='json', key_deid="deident", \
    key_ent="named_entities", key_start='charStartIdx', key_end='charEndIdx',
    key_subtype='label', key_text='text', key_type='type'):

    files = list(Path(source).glob('**/*.{}'.format(extension)))

    logging.info("")
    logging.info("Load automatic deid")
    logging.info("\tDirectory: {}".format(source))
    logging.info("\tFile count: {}".format(len(files)))


    d = OrderedDict()
    for file in files:

        with open(file, 'r') as f:
            deid_output = json.load(f)


        text = deid_output[key_text]


        entities = deid_output[key_deid][key_ent]

        id = str(file.relative_to(source).with_suffix(''))
        assert id not in d, '{} - {}'.format(d.keys(), id)

        #print('='*99)
        #print(repr(corpus[id].text()))
        #print('='*99)

        #print(repr(text))
        #print('='*99)
        assert corpus[id].text() == text



        d[id] = []
        for ent in entities:

            text_temp = text[ent[key_start]:ent[key_end]]

            a = ''.join(text_temp.split())
            b = ''.join(ent[key_text].split())
            assert a == b, "{} vs {}".format(a, b)

            entity = Entity( \
                type_ = ent[key_type],
                start = ent[key_start],
                end = ent[key_end],
                text = text_temp,
                subtype = ent[key_subtype],
            )

            d[id].append(entity)


    deid_summary(d, destination, "automatic")

    return d

def load_manual_deid(source, destination=None):

    logging.info("")
    logging.info("Load automatic deid")
    logging.info("\tCorpus source file: {}".format(source))

    corpus = joblib.load(source)

    d = OrderedDict()
    for doc in corpus.docs():
        id = os.sep.join(doc.id.split(os.sep)[1:])
        d[id] = doc.labels()

    logging.info("\tDocument count: {}".format(len(d)))

    deid_summary(d, destination, "manual")

    return d


def median_idx(x):
    return int(len(x) / 2)

def median_val(x):
    return x[int(len(x) / 2)]

def update_corpus(corpus, deid_dict, tokenizer, labels=LABELS, path=None):
    '''
    update corpus with deidentification dictionary
    '''

    logging.info("Update corpus")

    labels = [re.escape(label) for label in labels]

    # iterate over documents in corpus
    pbar = tqdm(total=len(corpus))
    match = 0
    before = Counter()
    after = Counter()
    for doc in corpus.docs():

        #only process if in deidentification dictionary
        #if (doc.id in deid_dict) and (doc.id == 'dev/uw/1300'):
        # if (doc.id in deid_dict) and (doc.id == 'dev/uw/2444'):


        #if (doc.id in deid_dict) and (doc.id == 'dev/uw/2630'):
        if (doc.id in deid_dict):

            match += 1
            for label in labels:
                before[label] += len(re.findall(label, doc.text()))

            # update document with entities
            entities = deid_dict[doc.id]
            update_doc(doc, entities, tokenizer)

            for label in labels:
                after[label] += len(re.findall(label, doc.text()))


        pbar.update()
    pbar.close()

    logging.info("\tCount with deid match: {}".format(match))

    a = []
    for k in before:
        a.append((k, before[k], after[k], after[k] - before[k]))

    df = pd.DataFrame(a, columns=["label", "before", "after", "change"])
    logging.info("\tCount deid:\n{}".format(df))


def sort_check_entities(entities):

    # sort entities in ascending order by indices
    entities = sorted(entities, key=lambda x: x.start, reverse=False)

    # make sure entities indices are increasing or not overlapping
    last = 0
    for entity in entities:
        assert entity.start >= last
        assert entity.end > entity.start
        last = entity.end

    return entities

def adjust_target_indices(original, new):



    if len(new) > len(original):
        #print('>')
        #print(original, len(original))
        #print(new, len(new))
        while len(new) > len(original):
            new.pop(median_idx(new))
        #print(new, len(new))
    elif len(new) < len(original):
        #print('<')
        #print(original, len(original))
        #print(new, len(new))

        while len(new) < len(original):
            med_val = median_val(new)
            new.insert(median_idx(new), med_val)
        #print(new, len(new))

    else:
        pass

    return new

def check_text(a, b):

    ws = lambda x: re.sub('\n', ' ', x)

    a = ws(a)
    b = ws(b)

    assert a == a, "'{}' vs '{}'".format(a, b)


def adjust_entities(entities, doc):

    # to get textbound indices
    splits = [x for _, tb in doc.tb_dict.items() for x in [tb.start, tb.end]]
    splits = sorted(list(set(splits)))

    # iterate over entities
    entities_out = []
    for entity in entities:

        # get all relevant splits from text bounds
        splits_temp =  [s for s in splits if \
                        (s > entity.start) and (s < entity.end)]

        # split deidentification entities, if spans occur in middle
        if len(splits_temp) > 0:

            # include end
            splits_temp.append(entity.end)

            # iterate over splits
            start = entity.start
            entities_temp = []
            for end in splits_temp:

                # create new entity as copy
                ent = deepcopy(entity)

                # get relative positions
                relative_start = start - entity.start
                relative_end = relative_start + (end - start)

                # get new text snippet
                text_new = entity.text[relative_start:relative_end]

                # remove leading and trailing whitespace
                ent.text, ent.start, ent.end = \
                                remove_white_space_at_ends(text_new, start, end)

                # save end position for next iteration
                start = end

                entities_temp.append(ent)

            assert entities_temp[0].start == entity.start
            assert entities_temp[-1].end == entity.end
            assert ''.join([''.join(ent.text.split()) for ent in entities_temp]) == ''.join(entity.text.split())

            entities_out.extend(entities_temp)

        else:
            entities_out.append(entity)

    return entities_out

def update_doc(doc, entities, tokenizer):
    '''
    Update document based on d identification entities
    '''
    # get current document text
    text = doc.text()

    #print('='*82)
    #print(text)
    #print('='*82)

    #print(repr(text))
    #print(repr(text[245:255]))
    #print('='*82)
    # document length, character count
    n = len(text)

    # sort entities and make sure non overlapping
    entities = sort_check_entities(entities)

    # get textbound indices,unique only,sort
    entities = adjust_entities(entities, doc)

    # iterate over the identification entities
    last = 0
    offset = 0
    char_map = OrderedDict()
    #text_new = []
    text_new = ''
    for entity in entities:

        check_text(text[entity.start:entity.end], entity.text)


        '''
        capture characters before entity start
        '''
        if last < entity.start:
            #print()
            #print('phase 1')
            text_new += text[last:entity.start]
            for i in range(last, entity.start):
                j = i + offset
                char_map[i] = j
                #print(i, j, text[i], text_new[j])

                assert text[i] == text_new[j], "{} vs {}".format(text[i], text_new[j])

        '''
        capture the character in entity
        '''
        #print('phase 2', entity.start, entity.end)
        entity_text_original = entity.text
        entity_text_new = DEID_PAT_LABEL.format(entity.subtype)

        length_original = len(entity_text_original)
        length_new = len(entity_text_new)

        indices_original = list(range(entity.start, entity.end))

        sub_start = entity.start + offset
        sub_end = sub_start + length_new

        indices_new = list(range(sub_start, sub_end))

        #text_new.append(entity_text_new)
        text_new += entity_text_new

        indices_new = adjust_target_indices(indices_original, indices_new)


        assert len(indices_original) == len(indices_new)
        for i, j in zip(indices_original, indices_new):
            char_map[i] = j
            #print(i, j, text[i], text_new[j])

        offset += length_new - length_original
        #print(repr(entity_text_original), len(entity_text_original))
        #print(length_new, length_original, offset)
        last = entity.end

    '''
    capture characters after last entity
    '''
    #print("phase 3")
    text_new += text[last:n]
    for i in range(last, n+1):
        j = i + offset
        char_map[i] = j
        if (i < len(text)) and (j < len(text_new)):
            #print(i, j, text[i], text_new[j])
            assert text[i] == text_new[j], "{} vs {}".format(text[i], text_new[j])

    doc.update_text(text_new, tokenizer, char_map)

def matcher_str(label, i1, i2, j1, j2, x, y):
    return '{:7}   x[{}:{}] --> y[{}:{}] {!r:>8} --> {!r}'.format( \
                    label, i1, i2, j1, j2, x[i1:i2], y[j1:j2])


def compare_text(A, B, doc_id):

    A = simple_tokenization(A)
    B = simple_tokenization(B)

    s = difflib.SequenceMatcher(None, A, B, autojunk=False)

    # iterator over aligned portions of strings
    counter = Counter()
    for label, i1, i2, j1, j2 in s.get_opcodes():

        if label != "equal":
            # print(matcher_str(label, i1, i2, j1, j2, a, b))

            a = ' '.join(A[i1:i2])
            b = ' '.join(B[j1:j2])
            counter[(doc_id, label, a, b)] += 1

    return counter

def compare_entities(A, B, doc_id):

    assert len(A) == len(B)

    counter = Counter()
    for id in A:
        a = A[id]
        b = B[id]

        assert a.type_ == b.type_
        assert a.subtype == b.subtype

        if a.text != b.text:
            counter[(doc_id, a.type_, a.subtype, a.text, b.text)] += 1

    return counter


def check_corpus_deidentification(corpus_a, corpus_b, path=None):

    docs_a = corpus_a.docs(out_type="dict")
    docs_b = corpus_b.docs(out_type="dict")

    assert len(docs_a) == len(docs_b)

    counter_text = Counter()
    counter_entities = Counter()
    for id in docs_a:
        doc_a = docs_a[id]
        doc_b = docs_b[id]

        counter_text += compare_text(doc_a.text(), doc_b.text(), doc_a.id)

        counter_entities += compare_entities(doc_a.entities(), doc_b.entities(), doc_a.id)


    x = [(id, label, b_, a_, count) for (id, label, a_, b_), count in counter_text.items()]
    df_text = pd.DataFrame(x, columns=["id", "label", "deid", "original", "count"])

    x = [(id, type_, subtype, b_, a_, count) for (id, type_, subtype, a_, b_), count in counter_entities.items()]
    df_entities = pd.DataFrame(x, columns=["id", "type", "subtype", "deid", "original", "count"])


    if path is not None:
        f = os.path.join(path, "check_text.csv")
        df_text = df_text.sort_values(["id", "deid", "count"], ascending=[True, True, False])
        df_text.to_csv(f, index=False)

        f = os.path.join(path, "check_entties.csv")
        df_entities = df_entities.sort_values(["id", "type", "count"], ascending=[True, True, False])
        df_entities.to_csv(f, index=False)

    return (df_text, df_entities)
