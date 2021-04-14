


import json
import os
import joblib
import re
import shutil
import pandas as pd
from multiprocessing import Pool
import traceback
from tqdm import tqdm
import numpy as np
from collections import OrderedDict, Counter
import logging
from pathlib import Path
import itertools

from corpus.tokenization import get_tokenizer
from corpus.document import Document
from config.constants import ENCODING, TEXT_FILE_EXT, ANN_FILE_EXT, QC
from corpus.brat import write_txt, write_ann
from utils.proj_setup import make_and_clear
from utils.random_sample import random_sample

def batch_documents(doc):
    '''
    Process batch of documents
    '''

    # Executed with error handling
    try:
        return Document(**doc)

    # Catch exceptions
    except Exception as e:
        print('Caught exception in worker thread:')

        for k, v in doc.items():
            print('{}:\t{}'.format(k, v))

        # Print exception
        traceback.print_exc()

        raise e




def include_keep(tags, include):

    # assume keep is true by default
    keep = True

    # exclude labels provided
    if (include is not None):

        # require all include tags to be present
        if not include.issubset(tags):
            keep = False

    return keep


def exclude_keep(tags, exclude):

    # assume keep is true by default
    keep = True

    # exclude labels provided
    if (exclude is not None):

        # at least some overlap between exclude and tags
        if len(exclude.intersection(tags)) > 0:
            keep = False

    return keep


class Corpus:
    '''
    Corpus container (collection of documents)
    '''
    def __init__(self):


        self.docs_ = OrderedDict()

    def __len__(self):
        return len(self.docs_)

    def __getitem__(self, key):
        return self.docs_[key]

    def __setitem__(self, key, item):
        self.docs_[key] = item

    def __delitem__(self, key):
        del self.docs_[key]

    def add_doc(self, doc):
        '''
        Add new document to corpus
        '''

        # Prevent duplicate document IDs
        assert doc.id not in self.docs_, \
        "corpus ids:\n{}\ndoc id:\t{}".format(self.docs_.keys(), doc.id)

        # Add document to corpus
        self.docs_[doc.id] = doc


        return True

    def doc_filter(self, include=None, exclude=[QC]):
        '''
        Get filtered set of documents
        '''

        if isinstance(include, str):
            include = [include]

        if isinstance(exclude, str):
            exclude = [exclude]


        if (include is not None):
            include = set(include)

        if (exclude is not None):
            exclude = set(exclude)

        docs_out = OrderedDict()
        for id, doc in self.docs_.items():

            # go to document tags
            tags = doc.tags
            if tags is None:
                tags = set([])
            if not isinstance(tags, set):
                tags = set(tags)

            keep = True
            keep = keep and include_keep(tags, include)
            keep = keep and exclude_keep(tags, exclude)

            if keep:
                docs_out[id] = doc

        if (include is not None) or (exclude is not None):
            logging.info('Document filter')
            logging.info('\tinclude:         {}'.format(include))
            logging.info('\texclude:         {}'.format(exclude))
            logging.info('\tcount, all:      {}'.format(len(self)))
            logging.info('\tcount, filtered: {}'.format(len(docs_out)))

        return docs_out


    def id2stem(self, id):
        '''
        Convert document ID to filename stem
        '''
        return id

    def docs(self, out_type='list', **kwargs):
        '''
        Get documents
        '''


        # Get filtered documents
        docs = self.doc_filter(**kwargs)

        # Output documents as dict (no change to output needed)
        if out_type == 'dict':
            pass

        # Output documents as list
        elif out_type == 'list':
            docs = [doc for k, doc in docs.items()]

        # Error case
        else:
            raise ValueError('''Invalid "out_type":\t{}'''.format(out_type))

        return docs


    def X(self, **kwargs):

        X = []
        for doc in self.docs(out_type='list', **kwargs):
            X.append(doc.X())

        return X

    def ids(self, as_stem=False, **kwargs):
        '''
        Get tokenized documents
        '''
        ids = []
        for doc in self.docs(out_type='list', **kwargs):

            id = doc.id
            if as_stem:
                id = self.id2stem(id)
            ids.append(id)

        return ids

    def tokens(self, **kwargs):
        '''
        Get tokenized documents
        '''
        tokens = []
        for doc in self.docs(out_type='list', **kwargs):
            tokens.append(doc.tokens())

        return tokens

    def doc_count(self, **kwargs):
        '''
        Get document count
        '''

        return len(self.docs(**kwargs))

    def sent_count(self, **kwargs):
        '''
        Get sentence count
        '''
        n = 0
        for doc in self.docs(**kwargs):
            n += doc.sent_count()
        return n

    def word_count(self, **kwargs):
        '''
        Get word count
        '''
        n = 0
        for doc in self.docs(**kwargs):
            n += doc.word_count()
        return n

    def counts(self, **kwargs):
        '''
        Get basic counts
        '''
        dc = self.doc_count(**kwargs)
        sc = self.sent_count(**kwargs)
        wc = self.word_count(**kwargs)


        columns = ['doc count', 'sent count', 'word count']
        df = pd.DataFrame([[dc, sc, wc]], columns=columns)

        return df

    def histogram(self, path=None, **kwargs):

        sent_lengths = Counter()
        doc_lengths = Counter()
        for doc in self.docs(out_type='list', **kwargs):
            tokens = doc.tokens()
            for sent in tokens:
                sent_lengths[len(sent)] += 1
            doc_lengths[len(tokens)] += 1

        df_sent = pd.DataFrame(sent_lengths.items(), columns=["sentence_length", "count"])
        df_sent.sort_values("sentence_length", ascending=True, inplace=True)

        df_doc = pd.DataFrame(doc_lengths.items(), columns=["document_length", "count"])
        df_doc.sort_values("document_length", ascending=True, inplace=True)

        if path is not None:
            f = os.path.join(path, "sentence_lengths.csv")
            df_sent.to_csv(f, index=False)

            f = os.path.join(path, "document_lengths.csv")
            df_doc.to_csv(f, index=False)

        return (df_sent, df_doc)

    def summary(self, path=None, **kwargs):
        '''
        Create corpus summary
        '''

        df = self.counts(**kwargs)

        logging.info('')
        logging.info('Corpus summary:\n{}'.format(df))
        logging.info('')

        if path is not None:
            fn = os.path.join(path, 'corpus_summary.csv')
            df.to_csv(fn)

        return df

    def write_examples(self, path, num_examples=50, flat_dir=True, **kwargs):
        '''
        Write examples
        '''

        # Iterate over documents
        for i, doc in enumerate(self.docs(**kwargs)):

            # Convert ID to file stem
            stem = self.id2stem(doc.id)

            # Flatten directory
            if flat_dir:
                stem = stem.replace('/', '_')

            fn = '{}.txt'.format(stem)
            fn = os.path.join(path, fn)
            Path(fn).parent.mkdir(parents=True, exist_ok=True)

            example = []
            example.append('='*72)
            example.append('Original')
            example.append('='*72)
            example.append(doc.text())

            example.append('='*72)
            example.append('Tokenized')
            example.append('='*72)
            example.append(doc.text_tokenized())

            with open(fn,'w') as f:
                f.write('\n'.join(example))

            if i >= num_examples:
                break

        return True


    def random_sample(self, size, \
            exclude = None,
            seed = 1,
            path = None,
            brat = True,
            encoding = ENCODING,
            footer = None,
            annotators = None,
            anno_type = 'single',  #  'single' or 'multiple'
            **kwargs):

        '''
        Randomly sample documents
        '''


        # Get relevant documents
        docs = self.docs(out_type='dict', **kwargs)

        # IDs as list
        ids = sorted(list(docs.keys()))

        ids_sample = random_sample(ids, size, seed=seed, exclude=exclude)

        # Extract samples
        docs_sample = OrderedDict()
        for id_, doc in docs.items():
            if id_ in ids_sample:
                docs_sample[id_] = doc
        assert len(docs_sample) == size

        # Write sampled files
        if not path is None:


            if (annotators is not None) and (anno_type == 'single'):
                annotators = itertools.cycle(annotators)


            # Loop on documents
            for id_, doc in docs_sample.items():

                stem = self.id2stem(id_)
                text = doc.text()
                if footer is not None:
                    text = "{}\n{}".format(text, footer)

                if anno_type == 'single':
                    annotator = next(annotators)

                    fn = os.path.join(annotator, stem)
                    write_txt(path, fn, text)
                    if brat:
                        write_ann(path, fn, '')


                elif anno_type == 'multiple':
                    for annotator in annotators:
                        fn = os.path.join(annotator, stem)
                        write_txt(path, fn, text)
                        if brat:
                            write_ann(path, fn, '')

                else:
                    ValueError("invalid annotation type: {}".format(anno_type))



        return docs_sample




    def export_text(self, path, **kwargs):
        '''
        Write tokenization example
        '''
        # Create directory does not exist
        if not os.path.exists(path):
            os.mkdir(path)

        # Loop on documents
        file_list = []
        for doc in self.docs(**kwargs):

            # Original text file
            fn = os.path.join(path, '{}.txt'.format(doc.id))
            file_list.append(fn)

            # recursively create directory structure
            parent = Path(fn).parent
            if not parent.exists():
                parent.mkdir(parents=True)

            with open(fn, 'w', encoding=ENCODING) as f:
                f.write(doc.text())

        return file_list

    def ids2tags(self, func, **kwargs):

        for doc in self.docs(**kwargs):
            if doc.tags is None:
                doc.tags = set([])

            for tag in func(doc.id):
                doc.tags.add(tag)


    def tokenization_examples(self, path, n, **kwargs):

        make_and_clear(path, recursive=True)


        for doc in self.docs(**kwargs)[:n]:

            # Output file name
            fn = os.path.join(path, '{}_original.{}'.format(doc.id, 'txt'))

            # Directory, including path in id
            dir_ = os.path.dirname(fn)
            if not os.path.exists(dir_):
                os.makedirs(dir_)

            with open(fn, 'w') as f:
                f.write(doc.text())

            fn = os.path.join(path, '{}_tokenized.{}'.format(doc.id, 'txt'))
            with open(fn, 'w') as f:
                f.write('\n====\n====\n'.join(doc.sents()))

    def write_ids(self, path, **kwargs):


        ids = []
        for id in self.docs(out_type='dict', **kwargs):
            ids.append(self.id2stem(id))


        fn = os.path.join(path, "ids.json")
        with open(fn, 'w') as f:
            json.dump(ids, f, indent=4)
