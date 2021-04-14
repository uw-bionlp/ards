



import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import logging
from torch.nn import ConstantPad3d, ConstantPad2d


from layers.utils import set_model_device, set_tensor_device

'''
tutorial4 tokenization
https://mccormickml.com/2019/07/22/BERT-fine-tuning/


how to use clinical bert
https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT


align ng character offsets with bert tokenization
https://github.com/LightTag/sequence-labeling-with-transformers/blob/master/sequence_aligner/dataset.py
'''

INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
OFFSET_MAPPING = 'offset_mapping'
PRETRAINED = "emilyalsentzer/Bio_ClinicalBERT"




def tokenize_documents(documents, \
    pretrained=PRETRAINED,
    add_special_tokens=True,
    max_length=50,
    return_attention_mask=True,
    return_tensors='pt',
    return_offsets_mapping=True,
    is_split_into_words=False
    ):

    logging.info("Tokenization using AutoTokenizer")

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained)


    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    mask = []
    offsets = []

    pbar = tqdm(total=len(documents))
    for i, text in enumerate(documents):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.

        encoded_dict = tokenizer.batch_encode_plus(
                    text,                      # Sentence to encode.
                    add_special_tokens = add_special_tokens, # Add '[CLS]' and '[SEP]'
                    max_length = max_length,           # Pad & truncate all sentences.
                    padding = 'max_length',
                    truncation = True,
                    return_attention_mask = return_attention_mask,   # Construct attn. masks.
                    return_tensors = return_tensors,     # Return pytorch tensors.
                    return_offsets_mapping = return_offsets_mapping,
                    is_split_into_words = is_split_into_words)

        input_ids.append(encoded_dict[INPUT_IDS])
        mask.append(encoded_dict[ATTENTION_MASK])

        offsets_ = encoded_dict[OFFSET_MAPPING].tolist()
        offsets_ = [[tuple(token) for token in sentence] for sentence in offsets_]
        offsets.append(offsets_)

        if i == 0:

            logging.info("-"*80)
            logging.info("")
            logging.info("Returned params:\n{}".format(encoded_dict.keys()))
            logging.info("")
            logging.info('Input:\n{}'.format(text))


            logging.info("")
            #logging.info('IDs: {}\n{}'.format(input_ids[0].shape, input_ids[0]))
            logging.info('IDs: {}'.format(input_ids[0].shape))

            logging.info("")
            #logging.info('Attn: {}\n{}'.format(mask[0].shape, mask[0]))
            logging.info('Attn: {}'.format(mask[0].shape))

            wps = [tokenizer.convert_ids_to_tokens(ids_) for ids_ in input_ids[0].squeeze()]
            logging.info("")
            logging.info('Tok:\n')
            for wps_ in wps[:10]:
                logging.info(f'{wps_[:10]} ....')

            #logging.info("")
            #logging.info('Idx:\n{}'.format(offsets[0]))
            #logging.info("")
            #logging.info("-"*80)

        pbar.update()
    pbar.close()

    logging.info("")
    logging.info('Document count: {}'.format(len(input_ids)))
    logging.info("")

    return (input_ids, mask, offsets)

def encode_documents(input_ids, mask, \
    pretrained=PRETRAINED,
    device=None,
    train=False):


    logging.info("Embedding using AutoModel")

    model = AutoModel.from_pretrained(pretrained)

    if train:
        model.train()
    else:
        model.eval()


    set_model_device(model, device)

    X = []
    masks = []
    pbar = tqdm(total=len(input_ids))
    assert len(input_ids) == len(mask)
    for i, (ids, msk) in enumerate(zip(input_ids, mask)):


        ids = set_tensor_device(ids, device)
        msk = set_tensor_device(msk, device)

        x = model( \
            ids,
            token_type_ids=None,
            attention_mask=msk)[0]

        x = x.cpu().detach()
        X.append(x)

        if i == 1:

            logging.info("Encode documents")

            #logging.info("-"*80)

            #logging.info("")
            #logging.info('IDs: {}\n{}'.format(ids.shape, ids))
            logging.info('IDs: {}'.format(ids.shape))

            #logging.info("")
            #logging.info('Mask: {}\n{}'.format(msk.shape, msk))
            logging.info('Mask: {}'.format(msk.shape))

            #logging.info("")
            #logging.info('X: {}\n{}'.format(x.shape, x))
            logging.info('X: {}'.format(x.shape))
            logging.info('')
            #logging.info("")
            #logging.info("-"*80)

    pbar.update()
    pbar.close()

    logging.info("")
    logging.info('Document count: {}'.format(len(X)))
    logging.info("")

    return X






def char2wordpiece(start, end, offsets):
    '''
    convert character indices to word piece indices
    (i.e. document)

    Parameters
    ----------
    char_indices: character indices for span
    offsets: offsets returned by transformer tokenizer

    Returns
    -------
    word_indices: word piece indices for spans
    '''


    start_new = -1
    end_new = -1
    for index, (start_word, end_word) in enumerate(offsets):
        # start_word = character index of word piece start (inclusive)
        # end_word = character index of word piece end (exclusive)
        # index = index of word peice in sentence

        if (start_new == -1) and \
           (start >= start_word) and \
           (start <  end_word):

            start_new = index

        if (end_new == -1) and \
           (end >  start_word) and \
           (end <= end_word):

           # add one so end_new is exclusive
           end_new = index + 1

    assert start_new != -1
    assert end_new != -1

    return (start_new, end_new)


def wordpiece2char(start, end, offsets):
    '''
    convert word piece indices to character indices for sequence of sentences
    (i.e. document)

    Parameters
    ----------
    word_indices: word piece indices for spans
    offsets: offsets returned by transformer tokenizer

    Returns
    -------
    char_indices: character indices per spans
    '''


    indices = offsets[start:end]

    # character index of start
    start_new = indices[0][0]

    # character index of end
    end_new = indices[-1][-1]

    return (start_new, end_new)




def demo():

    #loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    #for logger in loggers:#
    #    logger.setLevel(logging.info)

    documents = [['patient is reporting fever and cough.', 'chest x re indicates bilateral infile traits'],
                 ['diffuse lung disease', 'reporting position is addr']]

    tokens = tokenize_documents(documents, max_length=19)
    embedding = encode_documents(tokens)
