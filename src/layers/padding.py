
import torch
import logging
from torch.nn import ConstantPad3d, ConstantPad2d, ConstantPad1d



def pad3D(x, max_seq_count):
    '''
    pad first dimension (sequence count) of documents
    '''

    # current sentence count
    seq_count = x.shape[0]

    # append zeros, if sequence count too low
    if seq_count < max_seq_count:
        padding_back = max_seq_count - seq_count
        pad = ConstantPad3d((0, 0, 0, 0, 0, padding_back), 0)
        x = pad(x)

    # truncate document
    elif seq_count > max_seq_count:
        x = x[:max_seq_count]

    return x

def pad2D(x, max_seq_count):
    '''
    pad first dimension (sequence count) of documents
    '''

    # current sentence count
    seq_count = x.shape[0]

    # append zeros, if sequence count too low
    if seq_count < max_seq_count:
        padding_back = max_seq_count - seq_count
        pad = ConstantPad2d((0, 0, 0, padding_back), 0)
        x = pad(x)

    # truncate document
    elif seq_count > max_seq_count:
        x = x[:max_seq_count]

    return x


def pad1D(x, max_seq_count):
    '''
    pad first dimension (sequence count) of documents
    '''

    # current sentence count
    seq_count = x.shape[0]

    # append zeros, if sequence count too low
    if seq_count < max_seq_count:
        padding_back = max_seq_count - seq_count
        pad = ConstantPad1d((0, padding_back), 0)
        x = pad(x)

    # truncate document
    elif seq_count > max_seq_count:
        x = x[:max_seq_count]

    return x
