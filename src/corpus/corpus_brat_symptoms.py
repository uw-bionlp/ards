

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from collections import OrderedDict, Counter
import hashlib
import logging

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



from corpus.corpus_brat import CorpusBrat, counter2df
from corpus.document_brat_symptoms import DocumentBratSymptoms
from config.constants import TRAIN, DEV, TEST



class CorpusBratSymptoms(CorpusBrat):

    def __init__(self):

        CorpusBrat.__init__(self, document_object=DocumentBratSymptoms)
