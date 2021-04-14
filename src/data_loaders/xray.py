





import os
import json
from collections import OrderedDict, Counter
from operator import itemgetter
import logging
from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 100000)
import re
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

from corpus.tokenization import get_tokenizer
#from config.constants import *
from corpus.document import Document
from corpus.corpus_xray import CorpusXray
from corpus.corpus_emerge_xray import CorpusEmergeXray, DATE_FORMAT
from config.constants_pulmonary import NONE, PRESENT, UNILATERAL, BILATERAL


STUDY_ID = "study_id"
ACCESSION = 'accession'
REPORT = 'report'
DISCH_DAY = 999


Q1 = 'Q1_consolidation'
Q2 = 'Q2_consolidation'
Q3 = 'Q3_consolidation'
Q4 = 'Q4_consolidation'
LEFT_QUADRANTS = [Q1, Q2]
RIGHT_QUADRANTS = [Q3, Q4]
CONSOLIDATION_THRESHOLD = 1.0

ALL_ENCOUNTERS_WORKSHEET = "All encounters"
COLUMN_CORRECTION = ' \(extent\)'
DAY_PREFIX = "Day "


INFILTRATES_IMAGE = 'infiltrates_image'

# map scores worksheet names (day, rep)
#SCORES_DAY_MAP = OrderedDict()
#SCORES_DAY_MAP["Day 0"] = (0, 1)
#SCORES_DAY_MAP["Day 1"] = (1, 1)
#SCORES_DAY_MAP["Day 2"] = (2, 1)
#SCORES_DAY_MAP["Day 3"] = (3, 1)

#ACCESSION_COLUMNS = OrderedDict()
#ACCESSION_COLUMNS["Day 0"] = 'radiograph1_accession_d0'
#ACCESSION_COLUMNS["Day 1"] = 'radiograph1_accession_d1'
#ACCESSION_COLUMNS["Day 2"] = 'radiograph1_accession_d2'
#ACCESSION_COLUMNS["Day 3"] = 'radiograph1_accession_d3'




LF = r'\*CRLF\*'


# Map and counter columns to and day and rep (day, rep)
ALL_ENCOUNTERS_DAY_MAP = OrderedDict()
ALL_ENCOUNTERS_DAY_MAP["radiograph1_accession_d0"] = (0, 1)
ALL_ENCOUNTERS_DAY_MAP["radiograph1_accession_d1"] = (1, 1)
ALL_ENCOUNTERS_DAY_MAP["radiograph1_accession_d2"] = (2, 1)
ALL_ENCOUNTERS_DAY_MAP["radiograph1_accession_d3"] = (3, 1)
ALL_ENCOUNTERS_DAY_MAP["radiograph1_accession_d4"] = (4, 1)


#def save_columns(df, destination):
#
#    columns = df.columns.tolist()
#    fn = os.path.join(destination, 'spreadsheet_columns.json')
#    with open(fn,'w') as f:
#        json.dump(columns, f, indent=4)
#
#    return columns

def column_name_dissect(name):
    '''

    "radiograph1_accession_d1",
    "radiograph1_accession_disch",
    "radiograph1_report_d0",

    '''

    rep, typ_, day = name.split('_')

    rep = int(re.sub('radiograph', '', rep))

    if day == 'disch':
        day = DISCH_DAY
    elif re.match('d[0-9]{1,2}', day):
        day = int(day[1:])
    else:
        raise ValueError("invalid day:\t{}".format(day))

    return (rep, typ_, day)



def load_corpus(source, linebreak_bound=True):
    '''
    Load corpus
    '''

    logging.info('-'*72)
    logging.info('COVID-19 x-rays')
    logging.info('-'*72)
    logging.info('\tSpreadsheet location:\t{}'.format(source))

    # Initialize tokenizer
    tokenizer = get_tokenizer(linebreak_bound=linebreak_bound)

    # Read spreadsheet
    df = pd.read_excel(source)

    # Save columns
    #logging.info('\tColumns:\n{}'.format(df.columns.tolist()))

    # Instantiate corpus
    corpus = CorpusXray()
    accessions = OrderedDict()
    logging.info('Importing docs...')
    pbar = tqdm(total=len(df))

    # Iterate over rows (patients)
    for d in df.to_dict('records'):

        study_id = None



        # Loop on notes for current patient
        for k, v in d.items():

            if k == STUDY_ID:
                study_id = int(v)

            elif isinstance(v, str) or (not np.isnan(v)):

                rep, typ_, day = column_name_dissect(k)

                rep = int(rep)
                day = int(day)


                if typ_ == ACCESSION:
                    assert (study_id, day, rep) not in accessions
                    accessions[(study_id, day, rep)] = v

                elif typ_ == REPORT:

                    text_ = re.sub(LF, '\n', v)

                    doc = Document( \
                            id = (study_id, day, rep),
                            text_ = text_,
                            tokenizer = tokenizer)


                    corpus.add_doc(doc)

                else:
                    raise ValueError("could not resolve field")

        pbar.update(1)

    assert len(accessions) == len(corpus)

    for (study_id, day, rep), accession in accessions.items():
        corpus[(study_id, day, rep)].accession = accession

    return corpus


def load_emerge_corpus(source, linebreak_bound=True, \
        key_study_id='study_id',
        key_report_date='report_date',
        key_covid_test='before_covid_test',
        key_full_text='text_full',
        source_date_format="%Y-%m-%d %H:%M:%S"):


    '''
    Load corpus
    '''

    logging.info('-'*72)
    logging.info('eMERGE x-rays')
    logging.info('-'*72)
    logging.info('\tsource:\t{}'.format(source))

    # Initialize tokenizer
    tokenizer = get_tokenizer(linebreak_bound=linebreak_bound)

    # Load data
    data = json.load(open(source, 'r'))
    # Instantiate corpus
    corpus = CorpusEmergeXray()
    logging.info('Importing docs...')
    pbar = tqdm(total=len(data))


    # Iterate over rows (patients)
    keys = set([])
    for note in data:

        study_id = note[key_study_id]

        # convert to date information to datetime object
        date = note[key_report_date]
        date = date.split('.')[0]
        date = datetime.strptime(date, source_date_format)
        date = datetime.strftime(date, DATE_FORMAT)

        # define document id
        id = (study_id, date)

        # get covid information
        covid_test = note[key_covid_test]
        tags = set(['covid_test_{}'.format(covid_test)])

        text = note[key_full_text]

        if '\n' not in text:
            logging.warn(f"no linebreaks in text: {id}")

        doc = Document( \
                id = id,
                text_ = text,
                tokenizer = tokenizer,
                tags = tags)

        if id in keys:
            logging.warn(f"ID in corpus: {id}")
        else:
            corpus.add_doc(doc)
            keys.add(id)

        pbar.update(1)
    pbar.close()

    return corpus


def has_ards(x, threshold=1, q1=Q1, q2=Q2, q3=Q3, q4=Q4):

    left =  (x[q1] >= threshold) or (x[q2] >= threshold)
    right = (x[q3] >= threshold) or (x[q3] >= threshold)

    bilateral = left and right
    bilateral = int(bilateral)
    return bilateral


'''
def get_accession_map(workbook, \
        all_encounters_worksheet=ALL_ENCOUNTERS_WORKSHEET,
        all_encounters_day_map=ALL_ENCOUNTERS_DAY_MAP):

    #Get a map for accession values



    df = workbook[all_encounters_worksheet]

    map_ = OrderedDict()

    for d in df.to_dict(orient="records"):
        study_id = d[STUDY_ID]
        for column, (day, rep) in all_encounters_day_map.items():
            accession = d[column]

            k = (study_id, day, rep)
            assert k not in map_
            if np.isnan(accession):
                accession = None
            else:
                assert accession == float(int(accession))
                accession = int(accession)


            map_[k] = accession

    return map_
'''



def get_label(row, labels=[BILATERAL, UNILATERAL, PRESENT, NONE]):


    for label in labels:
        if (label in row) and (row[label] == 1):
            return label


def get_bilateral_infiltrate(df, \
        accession_key = ACCESSION,
        left_quadrants = LEFT_QUADRANTS,
        right_quadrants = RIGHT_QUADRANTS,
        threshold = CONSOLIDATION_THRESHOLD):

    df['left_score'] =  df[left_quadrants].max(axis=1)
    df['right_score'] = df[right_quadrants].max(axis=1)

    df['left_positive'] = df['left_score'].ge(threshold).astype(int)
    df['right_positive'] = df['right_score'].ge(threshold).astype(int)

    df[BILATERAL] = (df['left_positive'] + df['right_positive'] == 2).astype(int)
    df[UNILATERAL] = (df['left_positive'] + df['right_positive'] == 1).astype(int)
    df[NONE] =       (df['left_positive'] + df['right_positive'] == 0).astype(int)
    df[INFILTRATES_IMAGE] = df.apply(get_label, axis=1)

    return df

def load_xray_images(source, \
        day_prefix=DAY_PREFIX,

        q1=Q1, q2=Q2, q3=Q3, q4=Q4):

    # Load work book
    df = pd.read_csv(source)

    df = get_bilateral_infiltrate(df)

    return df
