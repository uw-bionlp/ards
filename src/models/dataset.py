

import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import logging
import numpy as np
from collections import Counter
import pandas as pd






class DatasetBase(Dataset):
    """

    """
    def __init__(self, X, y=None):
        super(DatasetBase, self).__init__()


    def __len__(self):
        NotImplementedError("not implemented")

    def __getitem__(self, index):
        NotImplementedError("not implemented")

    def preprocess_X(self, X):
        '''
        Preprocess X
        '''
        NotImplementedError("not implemented")

    def preprocess_y(self, y):
        '''
        Preprocess y
        '''
        NotImplementedError("not implemented")

    def postprocess_y(self):
        '''
        Postprocess y
        '''
        NotImplementedError("not implemented")


    def input_summary(self,):
        NotImplementedError("not implemented")
