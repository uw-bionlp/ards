

import os
import re
import pandas as pd
import numpy as np


from corpus.tokenization import get_tokenizer
from corpus.brat import write_txt





class Document:
    '''
    Document container
    '''
    def __init__(self, \
        id,
        text_,
        tags = None,
        patient = None,
        date = None,
        tokenizer = None,
        accession = None
        ):

        # Make sure text is not None and is string
        assert text_ is not None
        assert isinstance(text_, str)

        # Make sure text has at least 1 non-white space character
        text_wo_ws = ''.join(text_.split())
        assert len(text_wo_ws) > 0, '''"{}"'''.format(repr(text_))

        # Initialize tokenizer, if not provided
        #if tokenizer is None:
        #    tokenizer = get_tokenizer()

        # Store input text in spacy object
        self.spacy_obj = tokenizer(text_)

        # Other inputs
        self.id = id
        self.tags = tags
        self.patient = patient
        self.date = date
        self.accession = accession


    def __str__(self):
        return self.text()

    def update_text(self, text, tokenizer):
        self.spacy_obj = tokenizer(text)
        return True

    def text(self):
        '''
        Document text as string
        '''
        return self.spacy_obj.text

    def X(self):
        return self.text()

    def sents(self, keep_ws=True):
        '''
        Document text as list of sentence strings
        '''

        sents = []
        for sent in self.spacy_obj.sents:

            # Convert spacy ojb to string
            sent = str(sent)

            # Assess whether to keep sentence
            if keep_ws or (not sent.isspace()):
                sents.append(sent)

        return sents

    def sent_offsets(self):
        '''
        Document sentence offsets
        '''

        offsets = []
        #text = []
        #n = 0
        for sent in self.spacy_obj.sents:
            #sent_text = sent.text
            #text.append(sent_text)
            #rint(type(sent), len(sent), sent.start_char, sent.end_char, n)
            #print('"{}"'.format(repr(sent)))

            #start = sent.start
            #n += len(sent_text)


            #print(sent.start, sent.end, '"{}"'.format(repr(sent)))
            offsets.append((sent.start_char, sent.end_char))

        #text = "".join(text)
        #for a, b in zip(text, self.text()):
        #    print(repr(a), repr(b))
        #assert text == self.text(), '"{}" VS. "{}"'.format(len(text), len(self.text()))

        #z = sldfj
        return offsets


    def token_offsets(self, keep_ws=False):

        output = []
        for sent in self.spacy_obj.sents:
            for token in sent:
                text = token.text

                if keep_ws or (not text.isspace()):
                    start = token.idx
                    end = start + len(text)
                    output.append((text, start, end))

        return output

    def tokens(self, keep_ws=False):
        '''
        Get tokenized document as list of list
        '''
        doc_tokens = []

        # Loop on sentences
        for sent in self.spacy_obj.sents:

            # Loop on tokens in current sentence
            sent_tokens = []
            for tok in sent:

                # Convert spacy ojb to string
                tok = str(tok)

                # Assess whether the token is kept
                if keep_ws or (not tok.isspace()):
                    sent_tokens.append(tok)

            # Add tokenized sentence to document
            doc_tokens.append(sent_tokens)

        return doc_tokens

    def sent_count(self, keep_ws=False):
        '''
        Get sentence count
        '''
        n = 0
        for sent in self.sents(keep_ws=keep_ws):
            if keep_ws or (len("".join(sent.split())) > 0):
                n += 1
        return n

    def word_count(self, keep_ws=False):
        '''
        Get word count
        '''
        n = 0
        for sent in self.tokens(keep_ws=keep_ws):
            for tok in sent:
                n += 1
        return n


    def text_tokenized(self, **kwargs):
        '''
        Get tokenized text
        '''
        return "\n".join([str(sent) for sent in self.tokens(**kwargs)])

    def write_text(self, path):

        fn_text = write_txt(path, self.id, self.text())

        return fn_text
