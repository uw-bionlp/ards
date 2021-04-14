



import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import logging
from torch.nn import ConstantPad3d, ConstantPad2d
import copy
from math import ceil

INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
OFFSET_MAPPING = 'offset_mapping'
PRETRAINED = "emilyalsentzer/Bio_ClinicalBERT"

from corpus.tokenization import get_tokenizer
from layers.utils import set_model_device, set_tensor_device



def adjust_word_piece_offsets(sent_offsets, word_piece_offsets):
    '''

    '''

    n = len(sent_offsets)
    m = len(word_piece_offsets)
    assert n == m, f"{n} VS {m}"


    word_piece_offsets = copy.deepcopy(word_piece_offsets)
    for i, (sent_offsets, wp_offsets) in \
                           enumerate(zip(sent_offsets, word_piece_offsets)):
        sent_start, sent_end = sent_offsets
        for j, (wp_start, wp_end) in enumerate(wp_offsets):

            if (wp_start, wp_end) != (0,0):
                wp_start += sent_start
                wp_end += sent_start
                word_piece_offsets[i][j] = (wp_start, wp_end)

    return word_piece_offsets

def get_word_pieces_to_keep(token_offsets, word_piece_offsets, keep='last',
            pad_start=True, pad_end=True):

    assert len(token_offsets) == len(word_piece_offsets), f"{len(token_offsets)} VS {len(word_piece_offsets)}"
    for t, w in zip(token_offsets, word_piece_offsets):
        #assert len(t) <= len(w), f'''"{t}" VS "{w}"'''
        pass

    starts = []
    ends = []
    #to_keep = []
    for k, (token_seq, word_piece_seq) in enumerate(zip(token_offsets, word_piece_offsets)):
        starts.append([])
        ends.append([])


        for i, (tok_start, tok_end) in enumerate(token_seq):
            start_found = False
            end_found = False

            for j, (wp_start, wp_end) in enumerate(word_piece_seq):

                # account for start padding
                # k = j + int(pad_start)

                if (not start_found) and (tok_start >= wp_start) and (tok_start <  wp_end):
                    starts[-1].append(j)
                    start_found = True

                if (not end_found) and (tok_end > wp_start) and (tok_end <= wp_end):
                    ends[-1].append(j)
                    end_found = True

    to_keep = []
    assert len(starts) == len(ends), f"{len(starts)} VS {len(ends)}"
    for S, E in zip(starts, ends):

        assert len(S) == len(E), f"{len(S)} VS {len(E)}"

        if pad_start:
            S.insert(0,0)
            E.insert(0,0)
        if pad_end:
            e = E[-1]+1
            S.append(e)
            E.append(e)

        to_keep.append([])
        for s, e in zip(S, E):

            if keep == "first":
                to_keep[-1].append(s)
            elif keep == "last":
                to_keep[-1].append(e)
            elif keep == "mean":
                to_keep[-1].append((s, e + 1))

            else:
                raise ValueError(f"Invalid keep type: {keep}")

    return to_keep



def spacy_tokenize_doc(text, tokenizer, keep_ws=False, max_sent_count=None, \
                max_length=None, pad_start=False, pad_end=False):


    max_length_adj = max_length - int(pad_start) - int(pad_end)

    doc = tokenizer(text)

    sentences = []
    sent_offsets = []
    token_offsets = []
    for i, sent in enumerate(doc.sents):

        sent_start = sent.start_char
        sent_end =sent.end_char

        sent_offsets.append((sent_start, sent_end))
        sentences.append(str(sent))
        token_offsets.append([])

        if pad_start:
            token_offsets[-1].append((-1, -1))

        for j, token in enumerate(sent):
            token_text = token.text

            not_space = keep_ws or (not token_text.isspace())
            in_range = (max_length is None) or (j < max_length_adj)

            if not_space and in_range:
                token_start = token.idx
                token_end = token_start + len(token_text)
                token_offsets[-1].append((token_start, token_end))

        if pad_end:
            token_offsets[-1].append((-1, -1))


    if max_sent_count is not None:
        sentences = sentences[:max_sent_count]
        sent_offsets = sent_offsets[:max_sent_count]
        token_offsets = token_offsets[:max_sent_count]

    return (sentences, sent_offsets, token_offsets)




def wp_tokenize_doc(sentences, tokenizer, \
            add_special_tokens = True,
            max_length = 50,
            return_attention_mask = True,
            return_tensors = 'pt',
            return_offsets_mapping = True,
            is_split_into_words = False,
            verbose = False
            ):


    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.

    encoded_dict = tokenizer.batch_encode_plus(
                sentences,    # Sentence to encode.
                add_special_tokens = add_special_tokens, # Add '[CLS]' and '[SEP]'
                max_length = max_length,           # Pad & truncate all sentences.
                padding = 'max_length',
                truncation = True,
                return_attention_mask = return_attention_mask,   # Construct attn. masks.
                return_tensors = return_tensors,     # Return pytorch tensors.
                return_offsets_mapping = return_offsets_mapping,
                is_split_into_words = is_split_into_words)


    word_piece_offsets = encoded_dict[OFFSET_MAPPING].tolist()
    word_piece_offsets = [[tuple(token) for token in sentence] for sentence in word_piece_offsets]


    if verbose:
        logging.info('wp_tokenize_doc')
        for i, ids in enumerate(encoded_dict[INPUT_IDS]):
            wps = tokenizer.convert_ids_to_tokens(ids)[0:15]
            logging.info(f"{i}: {wps}")

    return (encoded_dict, word_piece_offsets)


def tokenize_documents(documents, max_length=None, max_sent_count=None,\
                    linebreak_bound=True, keep_ws=False, pad_start=True, pad_end=True):


    spacy_tokenizer = get_tokenizer(linebreak_bound=linebreak_bound)

    logging.info("Tokenizing documents...")

    sentences = []
    sent_offsets = []
    token_offsets = []
    pbar = tqdm(total=len(documents))
    for doc in documents:
        sent, sent_off, tok_off = spacy_tokenize_doc( \
                                        text = doc,
                                        tokenizer = spacy_tokenizer,
                                        keep_ws = keep_ws,
                                        max_sent_count = max_sent_count,
                                        max_length = max_length,
                                        pad_start = pad_start,
                                        pad_end = pad_end)

        sentences.append(sent)
        sent_offsets.append(sent_off)
        token_offsets.append(tok_off)

        pbar.update()
    pbar.close()
    return (sentences, sent_offsets, token_offsets)



def wp_tokenize_documents(sentences, sent_offsets, token_offsets, pretrained, max_wp_length=None,
                    keep='last', pad_start=True, pad_end=True, verbose=False):


    # Instantiate tokenizer
    wp_tokenizer = AutoTokenizer.from_pretrained(pretrained)

    logging.info("BERT Tokenizing documents...")

    encoded_dict = []
    word_piece_offsets = []
    word_pieces_keep = []
    pbar = tqdm(total=len(sentences))

    for sents, sent_off, tk_off in zip(sentences, sent_offsets, token_offsets):

        encoded, wp_offsets = wp_tokenize_doc( \
                                    sentences = sents,
                                    tokenizer = wp_tokenizer,
                                    max_length = max_wp_length)

        wp_offsets = adjust_word_piece_offsets( \
                                    sent_offsets = sent_off,
                                    word_piece_offsets = wp_offsets)

        wp_keep = get_word_pieces_to_keep( \
                                    token_offsets = tk_off,
                                    word_piece_offsets = wp_offsets,
                                    keep = keep,
                                    pad_start = pad_start,
                                    pad_end = pad_end)


        if verbose:
            logging.info('tokenize_documents')
            for i, (ids, kp) in enumerate(zip(encoded[INPUT_IDS], wp_keep)):
                wps = wp_tokenizer.convert_ids_to_tokens(ids)
                logging.info(f"{i}: {[wps[k] for k in kp]}")

        encoded_dict.append(encoded)
        word_piece_offsets.append(wp_offsets)
        word_pieces_keep.append(wp_keep)
        pbar.update()
    pbar.close()
    return (encoded_dict, word_pieces_keep)



def encode_document(encoded_dict, model, \
                        word_pieces_keep = None,
                        device = None,
                        detach = True,
                        move_to_cpu = True,
                        max_length = None,
                        verbose = False,
                        batch_size = 100):

    input_ids = encoded_dict[INPUT_IDS]
    mask = encoded_dict[ATTENTION_MASK]

    batches = ceil(len(mask)/batch_size)

    x_batches = []
    for i in range(batches):

        start = i*batch_size
        end = (i+1)*batch_size

        input_ids_batch = set_tensor_device(input_ids[start:end], device)
        mask_batch = set_tensor_device(mask[start:end], device)

        x = model( \
            input_ids = input_ids_batch,
            token_type_ids = None,
            attention_mask = mask_batch)[0]

        if move_to_cpu:
            x = x.cpu()

        if detach:
            x = x.detach()

        x_batches.append(x)

    x = torch.cat(x_batches, dim=0)
    assert len(x) == len(mask)


    if word_pieces_keep is not None:
        assert len(word_pieces_keep) == len(x)

        x_temp = torch.zeros_like(x)
        mask_temp = torch.zeros_like(mask)

        for i, wp_keep in enumerate(word_pieces_keep):
            for j, target in enumerate(wp_keep):
                if isinstance(target, (list, tuple)):
                    a, b = tuple(target)
                    x_temp[i, j, :] = x[i, a:b, :].mean(dim=0)
                    mask_temp[i, j] = mask[i, a]
                else:
                    x_temp[i, j, :] = x[i, target, :]
                    mask_temp[i, j] = mask[i, target]


        x = x_temp
        mask = mask_temp

    if max_length is not None:

        if word_pieces_keep is not None:
            assert x[:,max_length:,:].sum().tolist() == 0
            assert mask[:,max_length:].sum().tolist() == 0

        x = x[:,:max_length,:]
        mask = mask[:,:max_length]

    return (x, mask)


def encode_documents(encoded_dict, pretrained, \
            word_pieces_keep = None,
            device = None,
            train = False,
            detach = True,
            move_to_cpu = True,
            max_length = None):

    model = AutoModel.from_pretrained(pretrained)

    if train:
        model.train()
    else:
        model.eval()
    set_model_device(model, device)

    if word_pieces_keep is None:
        word_pieces_keep = [None for _ in encoded_dict]
    assert len(word_pieces_keep) == len(encoded_dict)

    logging.info("Encoding documents...")

    X = []
    mask =[]
    pbar = tqdm(total=len(encoded_dict))
    for encoded, wp_keep in zip(encoded_dict, word_pieces_keep):
        x, m = encode_document( \
                    encoded_dict = encoded,
                    model = model,
                    word_pieces_keep = wp_keep,
                    device = device,
                    detach = detach,
                    move_to_cpu = move_to_cpu,
                    max_length = max_length)

        X.append(x)
        mask.append(m)

        pbar.update()

    pbar.close()

    return (X, mask)



def get_ids_from_sentences(sentences, pretrained, max_length=50):

    tokenizer = AutoTokenizer.from_pretrained(pretrained)


    encoded_dict = []
    for i, sents in enumerate(sentences):
        d = tokenizer.batch_encode_plus(sents,
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = max_length,           # Pad & truncate all sentences.
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
            is_split_into_words = False)

        #if False: #i == 0:
        #    logging.info('wp_tokenize_doc')
        #    for i, ids in enumerate(encoded_dict[INPUT_IDS]):
        #        wps = tokenizer.convert_ids_to_tokens(ids)[0:15]
        #        logging.info(f"{i}: {wps}....")

        encoded_dict.append(d)

    return encoded_dict







def char2token_idx(start, end, offsets):
    '''
    convert character indices to token indices
    (i.e. document)

    Parameters
    ----------
    char_indices: character indices for span
    offsets: character offsets as list of list of tuple

    Returns
    -------
    (start_index, end_index): token indices associated with character indices
    '''

    sent_index = -1
    start_index = -1
    end_index = -1


    # iterate over sentences
    for i, offset_sent in enumerate(offsets):
        # i = index of sentence

        # iterate over tokens in sentence
        for j, (start_char, end_char) in enumerate(offset_sent):
            # start_char = character index of token start (inclusive)
            # end_char = character index of token end (exclusive)
            # j = index of token in sentence

            if (start >= start_char) and (start <  end_char):
                sent_index = i
                start_index = j
                break

    if sent_index == -1:
        logging.warn(f"Could not to map chars to tokens: {(start, end)}")
        return (None, None, None)


    # iterate over tokens in sentence
    for j, (start_char, end_char) in enumerate(offsets[sent_index]):
        if (end >  start_char) and (end <= end_char):
           # add one so end_index is exclusive
           end_index = j + 1
           break

    # set end token to last token, if none found
    if end_index == -1:
        last_char = -1

        for j, (start_char, end_char) in enumerate(offsets[sent_index]):
            if end_char >= last_char:
                last_char = end_char
                end_index = j + 1
        logging.warn(f"Truncating span: {(start, end)}")

    return (sent_index, start_index, end_index)



def token2char_idx(sent_index, start_index, end_index, offsets):
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

    assert sent_index < len(offsets), f"{sent_index} not in {offsets}"

    indices = offsets[sent_index][start_index:end_index]

    assert len(indices), f"sent {sent_index} and tokens ({start_index}, {end_index}) not in {offsets}"

    # character index of start
    start_new = indices[0][0]

    # character index of end
    end_new = indices[-1][-1]

    return (start_new, end_new)
