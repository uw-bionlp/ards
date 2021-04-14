import torch
import torch.nn as nn
from allennlp.modules import FeedForward
from allennlp.modules.time_distributed import TimeDistributed
from collections import OrderedDict
from allennlp.nn import util

import logging


NEG_FILL = -1e20



def logit_scorer(logits, agg_type='max'):
    '''
    Convert logits to single score
    '''
    if agg_type == 'max':
        scores, _ = logits[:,:,1:].max(dim=-1)
    elif agg_type == 'sum':
        scores = logits[:,:,1:].sum(dim=-1)
    else:
        raise ValueError("incorrect agg_type: {}".format(agg_type))
    return scores




def span_pruner(embeddings, scores, mask, seq_length, spans_per_word=1, num_keep=None):
        """

        Based on AllenNLP allennlp.modules.Pruner from release 0.84


        Parameters
        ----------

        logits: (batch_size, num_spans, num_tags)
        mask: (batch_size, num_spans)
        num_keep: int OR torch.LongTensor
                If a tensor of shape (batch_size), specifies the
                number of items to keep for each
                individual sentence in minibatch.
                If an int, keep the same number of items for all sentences.


        """


        #batch_size, num_items, num_tags = tuple(logits.shape)
        batch_size, num_items = tuple(scores.shape)


        # Number to keep not provided, so use spans per word
        if num_keep is None:
            num_keep = seq_length*spans_per_word
            num_keep = torch.max(num_keep, torch.ones_like(num_keep))



        # If an int was given for number of items to keep, construct tensor by repeating the value.
        if isinstance(num_keep, int):
            num_keep = num_keep*torch.ones([batch_size], dtype=torch.long,
                                                               device=mask.device)

        # Maximum number to keep
        max_keep = num_keep.max()

        # Get scores from logits
        # (batch_size, num_spans)
        # scores = logit_scorer(logits)

        # Set overlapping span scores large neg number
        #if prune_overlapping:
        #    scores = overlap_filter(scores, span_overlaps)

        # Add dimension
        scores = scores.unsqueeze(-1)
        #embeddings = embeddings.unsqueeze(-1)


        # Check scores dimensionality
        if scores.size(-1) != 1 or scores.dim() != 3:
            raise ValueError(f"The scorer passed to Pruner must produce a tensor of shape"
                             f"(batch_size, num_items, 1), but found shape {scores.size()}")

        # Make sure that we don't select any masked items by setting their scores to be very
        # negative.  These are logits, typically, so -1e20 should be plenty negative.
        #print("scores", scores.shape)
        #print('mask', mask.shape)
        mask = mask.unsqueeze(-1).bool()  #type(torch.BoolTensor)
        #print('mask', mask.shape, mask.type)
        scores = util.replace_masked_values(scores, mask, NEG_FILL)

        # Shape: (batch_size, max_num_items_to_keep, 1)
        _, top_indices = scores.topk(max_keep, 1)

        # Mask based on number of items to keep for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices_mask = util.get_mask_from_sequence_lengths(num_keep, max_keep)
        top_indices_mask = top_indices_mask.bool()

        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = top_indices.squeeze(-1)

        # Fill all masked indices with largest "top" index for that sentence, so that all masked
        # indices will be sorted to the end.
        # Shape: (batch_size, 1)
        fill_value, _ = top_indices.max(dim=1)
        fill_value = fill_value.unsqueeze(-1)
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = torch.where(top_indices_mask, top_indices, fill_value)
        # Now we order the selected indices in increasing order with
        # respect to their indices (and hence, with respect to the
        # order they originally appeared in the ``embeddings`` tensor).
        top_indices, _ = torch.sort(top_indices, 1)

        # Shape: (batch_size * max_num_items_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select items for each element in the batch.
        flat_indices = util.flatten_and_batch_shift_indices(top_indices, num_items)

        # Combine the masks on spans that are out-of-bounds, and the mask on spans that are outside
        # the top k for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        sequence_mask = util.batched_index_select(mask, top_indices, flat_indices)
        sequence_mask = sequence_mask.squeeze(-1).bool()
        top_mask = top_indices_mask & sequence_mask
        top_mask = top_mask.long()

        # Shape: (batch_size, max_num_items_to_keep, 1)
        top_scores = util.batched_index_select(scores, top_indices, flat_indices)
        top_embeddings = util.batched_index_select(embeddings, top_indices, flat_indices)

        # Shape: (batch_size, max_num_items_to_keep)
        top_scores = top_scores.squeeze(-1)
        #top_embeddings = top_embeddings.squeeze(-1)


        return (top_indices, top_embeddings, top_scores, top_mask)



def span_pruner2(scores, mask, seq_length, spans_per_word=1, num_keep=None):
        """

        Based on AllenNLP allennlp.modules.Pruner from release 0.84


        Parameters
        ----------

        logits: (batch_size, num_spans, num_tags)
        mask: (batch_size, num_spans)
        num_keep: int OR torch.LongTensor
                If a tensor of shape (batch_size), specifies the
                number of items to keep for each
                individual sentence in minibatch.
                If an int, keep the same number of items for all sentences.


        """


        #batch_size, num_items, num_tags = tuple(logits.shape)
        batch_size, num_items = tuple(scores.shape)


        # Number to keep not provided, so use spans per word
        if num_keep is None:
            num_keep = seq_length*spans_per_word
            num_keep = torch.clamp(num_keep, max=num_items)

            num_keep = torch.max(num_keep, torch.ones_like(num_keep))

        # If an int was given for number of items to keep, construct tensor by repeating the value.
        if isinstance(num_keep, int):
            num_keep = num_keep*torch.ones([batch_size], dtype=torch.long,
                                                               device=mask.device)

        # Maximum number to keep
        max_keep = num_keep.max()

        # Add dimension
        scores = scores.unsqueeze(-1)


        # Check scores dimensionality
        if scores.size(-1) != 1 or scores.dim() != 3:
            raise ValueError(f"The scorer passed to Pruner must produce a tensor of shape"
                             f"(batch_size, num_items, 1), but found shape {scores.size()}")

        # Make sure that we don't select any masked items by setting their scores to be very
        # negative.  These are logits, typically, so -1e20 should be plenty negative.
        #print("scores", scores.shape)
        #print('mask', mask.shape)
        mask = mask.unsqueeze(-1).bool()  #type(torch.BoolTensor)
        #print('mask', mask.shape, mask.type)
        scores = util.replace_masked_values(scores, mask, NEG_FILL)

        # Shape: (batch_size, max_num_items_to_keep, 1)
        _, top_indices = scores.topk(max_keep, 1)

        # Mask based on number of items to keep for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices_mask = util.get_mask_from_sequence_lengths(num_keep, max_keep)
        top_indices_mask = top_indices_mask.bool()

        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = top_indices.squeeze(-1)

        # Fill all masked indices with largest "top" index for that sentence, so that all masked
        # indices will be sorted to the end.
        # Shape: (batch_size, 1)
        fill_value, _ = top_indices.max(dim=1)
        fill_value = fill_value.unsqueeze(-1)
        # Shape: (batch_size, max_num_items_to_keep)
        top_indices = torch.where(top_indices_mask, top_indices, fill_value)
        # Now we order the selected indices in increasing order with
        # respect to their indices (and hence, with respect to the
        # order they originally appeared in the ``embeddings`` tensor).
        top_indices, _ = torch.sort(top_indices, 1)

        # Shape: (batch_size * max_num_items_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select items for each element in the batch.
        top_indices_flat = util.flatten_and_batch_shift_indices(top_indices, num_items)

        # Combine the masks on spans that are out-of-bounds, and the mask on spans that are outside
        # the top k for each sentence.
        # Shape: (batch_size, max_num_items_to_keep)
        sequence_mask = util.batched_index_select(mask, top_indices, top_indices_flat)
        sequence_mask = sequence_mask.squeeze(-1).bool()
        top_mask = top_indices_mask & sequence_mask
        top_mask = top_mask.long()

        # Shape: (batch_size, max_num_items_to_keep, 1)
        top_scores = util.batched_index_select(scores, top_indices, top_indices_flat)

        # Shape: (batch_size, max_num_items_to_keep)
        top_scores = top_scores.squeeze(-1)


        return (top_indices, top_indices_flat, top_scores, top_mask)



class SpanPruner(nn.Module):
    '''
    Span scorer


    Parameters
    ----------
    num_tags: label vocab size


    Returns
    -------
    arg_scores: tensor of scores (batch_size, trig_num, arg_num, num_tags)

    '''
    def __init__(self, spans_per_word=1, num_keep=None):
        super(SpanPruner, self).__init__()

        self.spans_per_word = spans_per_word
        self.num_keep = num_keep

        # softmax across last dimension (label dimension)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, embeddings, span_scores, span_mask, seq_length, verbose=False):


        probabilities = []
        for k, scores in span_scores.items():

            # (sequence_count, span_count, label_count)
            # scores

            # calculate probabilities from logits
            # (sequence_count, span_count, label_count)
            probs = self.softmax(scores)

            # remove null label probability
            # (sequence_count, span_count, label_count-1)
            probs = probs[:,:,1:]

            # find highest non null probability for type
            # (sequence_count, span_count)
            probs, _ = torch.max(probs, dim=2)

            probabilities.append(probs)

        # aggregate across types
        # (sequence_count, span_count, type_count)
        probabilities = torch.stack(probabilities, dim=2)

        # calculate maximum non null probability across types --> becomes the span score
        # (sequence_count, span_count)
        probabilities, _ = torch.max(probabilities, dim=2)

        # multiply by span mask to force invalid spans to have 0 probability of non null label
        #probabilities = probabilities*span_mask
        # print(probabilities.shape)

        # prune based on maximum of non null probabilities
        top_indices, top_embeddings, top_scores, top_mask = span_pruner( \
                                embeddings = embeddings,
                                scores = probabilities,
                                mask = span_mask,
                                seq_length = seq_length,
                                spans_per_word = self.spans_per_word,
                                num_keep = self.num_keep)

        if verbose:
            logging.info("")
            logging.info("SpanPruner")
            logging.info(f"\tprobabilities:  {probabilities.shape}")
            logging.info(f"\ttop_indices:    {top_indices.shape}")
            logging.info(f"\ttop_embeddings: {top_embeddings.shape}")
            logging.info(f"\ttop_scores:     {top_scores.shape}")
            logging.info(f"\ttop_mask:       {top_mask.shape}")


        return (top_indices, top_embeddings, top_scores, top_mask)



class SpanPrunerMulti(nn.Module):
    '''
    Span scorer


    Parameters
    ----------
    num_tags: label vocab size


    Returns
    -------
    arg_scores: tensor of scores (batch_size, trig_num, arg_num, num_tags)

    '''
    def __init__(self, spans_per_word=1, num_keep=None):
        super(SpanPrunerMulti, self).__init__()

        self.spans_per_word = spans_per_word
        self.num_keep = num_keep

        # softmax across last dimension (label dimension)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, embeddings, span_scores, span_mask, seq_length, verbose=False):



        assert isinstance(embeddings, dict)
        assert isinstance(span_scores, dict)


        probabilities = []
        for k, scores in span_scores.items():

            # (sequence_count, span_count, label_count)
            # scores

            # calculate probabilities from logits
            # (sequence_count, span_count, label_count)
            probs = self.softmax(scores)

            # remove null label probability
            # (sequence_count, span_count, label_count-1)
            probs = probs[:,:,1:]

            # find highest non null probability for type
            # (sequence_count, span_count)
            probs, _ = torch.max(probs, dim=2)

            probabilities.append(probs)

        # aggregate across types
        # (sequence_count, span_count, type_count)
        probabilities = torch.stack(probabilities, dim=2)

        # calculate maximum non null probability across types --> becomes the span score
        # (sequence_count, span_count)
        probabilities, _ = torch.max(probabilities, dim=2)

        # multiply by span mask to force invalid spans to have 0 probability of non null label
        #probabilities = probabilities*span_mask
        # print(probabilities.shape)

        # prune based on maximum of non null probabilities
        top_indices, top_indices_flat, top_scores, top_mask = span_pruner2( \
                                scores = probabilities,
                                mask = span_mask,
                                seq_length = seq_length,
                                spans_per_word = self.spans_per_word,
                                num_keep = self.num_keep)

        top_embeddings = OrderedDict()
        for k, embed in embeddings.items():
            top_embeddings[k] = util.batched_index_select(embed, top_indices, top_indices_flat)


        if verbose:
            logging.info("")
            logging.info("SpanPruner")
            logging.info(f"\tprobabilities:  {probabilities.shape}")
            logging.info(f"\ttop_indices:    {top_indices.shape}")
            for k, v in top_embeddings.items():
                logging.info(f"\ttop_embeddings: {v.shape}")
            logging.info(f"\ttop_scores:     {top_scores.shape}")
            logging.info(f"\ttop_mask:       {top_mask.shape}")


        return (top_indices, top_embeddings, top_scores, top_mask)
