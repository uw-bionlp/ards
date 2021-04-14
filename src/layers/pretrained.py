



import torch
import os
import json
import joblib
import logging

from config.constants import PARAMS_FILE, STATE_DICT


def load_pretrained(model_class, model_dir, param_map=None):
    '''
    Load pretrained Pytorch model
    '''


    logging.info('')
    logging.info('-'*72)
    logging.info('Loading pre-trained model from:\t{}'.format(model_dir))
    logging.info('-'*72)

    # Load hyper parameters
    f = os.path.join(model_dir, PARAMS_FILE)
    logging.info("\tParameters file:\t{}".format(f))
    parameters = joblib.load(f)

    # Print hyper parameters
    logging.info("\tParameters loaded:")
    for param, val in parameters.items():
        if isinstance(val, dict):
            logging.info(f"\t\t{param}")
            for k, v in val.items():
                logging.info(f"\t\t\t{k}:\t{v}")
        else:
            logging.info(f"\t\t{param}:\t{val}")
    logging.info('')

    # Map parameters
    if param_map is not None:
        logging.info('\tMapping parameters:')
        for orig, new_ in param_map.items():
            logging.info('\t\t{}: orig={},\tnew={}'.format(orig, parameters[orig], new_))
            #parameters[orig] = new_
            parameters[new_] = parameters[orig]
            del parameters[orig]

    # Load saved estimator
    fn = os.path.join(model_dir, STATE_DICT)
    logging.info("\tState dict file:\t{}".format(fn))
    state_dict = torch.load(fn, map_location=lambda storage, loc: storage)
    logging.info("\tState dict loaded")

    # Instantiate model
    model = model_class(**parameters)
    model.load_state_dict(state_dict)
    logging.info("\tModel instantiated and state dict loaded")
    logging.info('')

    return model
