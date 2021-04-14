import torch
import torch.nn as nn
import logging
from collections import Counter, OrderedDict
from torch.nn.parameter import Parameter


from modules.span_embedder import SpanEmbedderMulti, SpanEmbedder
from modules.span_scorer import SpanScorerMulti
from modules.span_scorer_crf import SpanScorerCRFMulti
from modules.span_pruner import SpanPruner, SpanPrunerMulti
from modules.role_scorer import RoleScorerMulti, RoleScorer, get_role_mask
from layers.utils import get_loss, aggregate
from layers.attention import Attention
from config.constants import ENTITIES, RELATIONS, DOC_LABELS
from layers.utils import get_predictions


def get_seq_lenth(mask, indices):
    # mask (batch size, span count)
    # indices (batch_size, span_count, 2)

    # get dimensionality
    seq_count, span_count = tuple(mask.shape)

    # get end index
    # (batch_size, span_count)
    end_indices = indices[:, :, 1]

    # reconfigure indices for multiplication
    #indices = indices[:,1].unsqueeze(0).repeat(seq_count, 1)

    # get largest valid indices
    # (batch_size)
    seq_length = torch.max(mask*end_indices, dim=1)[0]

    return seq_length

class RelationExtractor(nn.Module):
    '''

    '''
    def __init__(self, \
            entity_definition,
            relation_definition,
            input_dim,

            # span embedding
            span_embed_project = True,
            span_embed_dim = 100,
            span_embed_activation = "tanh",
            span_embed_dropout = 0.0,

            # span scoring
            span_scorer_type = "span",
            span_scorer_hidden_dim = 100,
            span_scorer_activation = 'relu',
            span_scorer_dropout = 0.0,
            span_class_weights = None,

            # span pruning
            spans_per_word = 1,

            # role scoring
            role_hidden_dim = 100,
            role_output_dim = 2,
            role_activation = 'relu',
            role_dropout = 0.0,

            create_doc_vector = False,
            doc_attention_dropout = 0.0,

            loss_reduction = "sum"

            ):
        super(RelationExtractor, self).__init__()

        self.loss_reduction = loss_reduction
        self.create_doc_vector = create_doc_vector

        self.embedders = SpanEmbedderMulti( \
                label_definition = entity_definition,
                input_dim = input_dim,
                project = span_embed_project,
                hidden_dim = span_embed_dim,
                activation = span_embed_activation,
                dropout = span_embed_dropout)

        self.span_scorer_type = span_scorer_type
        if self.span_scorer_type == "span":
            self.span_scorers = SpanScorerMulti( \
                    label_definition = entity_definition,
                    input_dim = self.embedders.output_dim,
                    hidden_dim = span_scorer_hidden_dim,
                    activation = span_scorer_activation,
                    dropout = span_scorer_dropout,
                    loss_reduction = loss_reduction,
                    class_weights = span_class_weights)

        elif self.span_scorer_type == "crf":
            self.span_scorers = SpanScorerCRFMulti( \
                    label_definition = entity_definition,
                    input_dim = input_dim,
                    loss_reduction = loss_reduction)

        else:
            raise ValueError("Invalid span scorer type")

        self.span_pruners = SpanPrunerMulti( \
                spans_per_word = spans_per_word)

        self.role_scorers = RoleScorerMulti( \
                label_definition = relation_definition,
                input_dim = self.embedders.output_dim*2,
                hidden_dim = role_hidden_dim,
                output_dim = role_output_dim,
                activation = role_activation,
                dropout = role_dropout,
                loss_reduction = loss_reduction
                )

        if self.create_doc_vector:
            self.doc_attention = Attention( \
                            input_dim = self.embedders.output_dim*2,
                            dropout = doc_attention_dropout,
                            use_ffnn = False,
                            activation = 'tanh')

            self.doc_init =  Parameter(torch.zeros((1, self.embedders.output_dim*2)))

        self.span_types = self.span_scorers.types

    def forward(self, seq_tensor, span_indices, span_mask, seq_mask=None, span_map=None, verbose=False, as_dict=False):
        '''

        Parameters
        ----------
        seq_tensor: sequence tensor (sequence count, sequence length, embedding dimension)
        seq_length: sequence tensor (sequence count, sequence length)
        span_indices: span indices (span count, 2)
        span_mask: span mask (sequence count, span count)


        '''


        # dict of tensors (sequence_count, span_count, embedding_dimension)
        embeddings = self.embedders( \
                                            seq_tensor = seq_tensor,
                                            span_indices = span_indices,
                                            span_mask = span_mask,
                                            verbose = verbose)


        if self.span_scorer_type == "span":

            # dict of tensors (sequence_count, span_count, label_count)
            span_scores = self.span_scorers(embeddings, span_mask, verbose=verbose)
            seq_scores = None

        elif self.span_scorer_type == "crf":
            seq_scores, span_scores = self.span_scorers( \
                                            seq_tensor = seq_tensor,
                                            seq_mask = seq_mask,
                                            span_map = span_map,
                                            span_indices = span_indices)


        seq_length = get_seq_lenth(span_mask, span_indices)
        top_indices, top_embeddings, top_span_scores, top_span_mask = self.span_pruners( \
                                        embeddings = embeddings,
                                        span_scores = span_scores,
                                        span_mask = span_mask,
                                        seq_length = seq_length,
                                        verbose = verbose)


        top_role_scores, embedding_pairs = self.role_scorers( \
                                            embeddings = top_embeddings,
                                            include_embed = True,
                                            verbose = verbose)





        if self.create_doc_vector:

            role_mask = get_role_mask(top_span_mask)
            role_embed = []
            for k in top_role_scores:

                scores = top_role_scores[k]
                embed = embedding_pairs[k]
                _, label_indices = scores.max(-1)
                label_pos = (label_indices > 0).to(scores.device)
                keep_bool = label_pos*role_mask
                keep_bool = keep_bool.bool().view(-1, 1).squeeze(-1)
                embed_pos = embed[keep_bool]

                role_embed.append(embed_pos)

            if len(role_embed):
                role_embed = [self.doc_init]



            role_embed = torch.cat(role_embed, dim=0).unsqueeze(0)
            doc_vector, doc_alphas = self.doc_attention(role_embed)

            doc_vector = doc_vector.squeeze(0)
            doc_alphas = doc_alphas.squeeze(0)

        else:
            doc_vector = None

        if as_dict:
            d = OrderedDict()
            d['span_scores'] = span_scores
            d['top_role_scores'] = top_role_scores
            d['top_span_mask'] = top_span_mask
            d['top_indices'] = top_indices
            d['doc_vector'] = doc_vector

            return d
        else:
            return (span_scores, top_role_scores, top_span_mask, top_indices)



    def loss(self, span_labels, span_scores, span_mask, role_labels, \
                                top_role_scores, top_span_mask, top_indices, seq_scores=None, seq_mask=None, span_map=None):

        if self.span_scorer_type == "span":
            span_loss = self.span_scorers.loss( \
                                        labels = span_labels,
                                        scores = span_scores,
                                        mask = span_mask)
        elif self.span_scorer_type == "crf":
            span_loss = self.span_scorers.loss( \
                                        span_labels = span_labels,
                                        seq_scores = seq_scores,
                                        seq_mask = seq_mask,
                                        span_map = span_map)

        role_loss = self.role_scorers.loss( \
                                    labels = role_labels,
                                    top_scores = top_role_scores,
                                    top_mask = top_span_mask,
                                    top_indices = top_indices)

        return (span_loss, role_loss)


    def perf_counts(self, span_labels, span_scores, span_mask, role_labels, \
                                top_role_scores, top_span_mask, top_indices):


        span_counts = self.span_scorers.perf_counts( \
                                    labels = span_labels,
                                    scores = span_scores,
                                    mask = span_mask)

        role_counts = self.role_scorers.perf_counts( \
                                    labels = role_labels,
                                    top_scores = top_role_scores,
                                    top_mask = top_span_mask,
                                    top_indices = top_indices)

        return (span_counts, role_counts)


    '''
    def prf(self, span_labels, span_scores, span_mask, role_labels, \
                                top_role_scores, top_span_mask, top_indices):


        span_prf = self.span_scorers.prf( \
                                    labels = span_labels,
                                    scores = span_scores,
                                    mask = span_mask)

        role_prf = self.role_scorers.prf( \
                                    labels = role_labels,
                                    top_scores = top_role_scores,
                                    top_mask = top_span_mask,
                                    top_indices = top_indices)

        return (span_prf, role_prf)
    '''
