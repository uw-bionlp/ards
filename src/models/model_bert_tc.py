
from collections import Counter, OrderedDict
import torch
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader

from allennlp.modules import FeedForward
from allennlp.nn.activations import Activation

from transformers import AutoTokenizer, AutoModel
from transformers import AdamW

import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import joblib
import math
import os




from layers.activation import get_activation
from models.model import Model, TUNE, PREDICT
from layers.attention import Attention
from layers.recurrent import RNN
from modules.relation_extractor import RelationExtractor
from modules.hierarchical_document_classifier import HierarchicalClassifierMulti, SentClassifiers
from layers.utils import set_model_device, set_tensor_device
from models.dataset_bert_tc import DatasetBertTC
from config.constants_pulmonary import INFILTRATES, EXTRAPARENCHYMAL
from config.constants import ENTITIES, RELATIONS, DOC_LABELS
from config.constants import PARAMS_FILE, STATE_DICT, PREDICTIONS_FILE
from layers.utils import get_loss, aggregate, PRFAggregator
from layers.plotting import PlotLoss
from scoring.scorer_xray import ScorerXray
from utils.misc import nest_list, nest_dict





class ModelBertTC(Model):

    def __init__(self,  \

        doc_definition,
        sent_definition,
        pretrained,
        num_workers,
        num_epochs,
        dropout_sent = 0.0,
        dropout_doc = 0.0,
        use_sent_objective = True,
        concat_sent_scores = True,
        dataset_class = DatasetBertTC,
        scorer_class = ScorerXray,
        grad_max_norm = 1.0,
        loss_reduction = 'sum',
        batch_size = 5,
        lr = 1e-5,
        lr_ratio = 1.0,
        attention_query_dim = 100,
        max_length = 50,
        max_sent_count = 50,
        linebreak_bound = True,
        keep_ws = False,
        project_sent = False,
        project_size = 200,
        optimizer_params = None,
        dataloader_params = None,
        hyperparams = None,
        dataset_params = None,


        ):

        super(ModelBertTC, self).__init__( \
            hyperparams = hyperparams,
            dataset_params = dataset_params,
            dataloader_params = dataloader_params,
            optimizer_params = optimizer_params,
            num_workers = num_workers,
            num_epochs = num_epochs,
            dataset_class = dataset_class,
            scorer_class = scorer_class
            )



        self.pretrained = pretrained
        self.use_sent_objective = use_sent_objective
        self.concat_sent_scores = concat_sent_scores
        self.grad_max_norm = grad_max_norm
        self.loss_reduction = loss_reduction

        self.doc_definition = doc_definition
        self.sent_definition = sent_definition
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.lr = lr
        self.lr_ratio = lr_ratio
        self.max_length = max_length
        self.max_sent_count = max_sent_count

        self.linebreak_bound = linebreak_bound
        self.keep_ws = keep_ws

        self.project_sent = project_sent
        self.project_size = project_size

        if self.concat_sent_scores:
            assert self.use_sent_objective



        self.bert = AutoModel.from_pretrained(self.pretrained)


        hidden_size = self.bert.config.hidden_size

        self.sent_attention = nn.ModuleDict(OrderedDict())
        self.doc_output_layers = nn.ModuleDict(OrderedDict())
        self.sent_ffnn = nn.ModuleDict(OrderedDict())
        self.sent_classifiers = nn.ModuleDict(OrderedDict())

        for k, label_set in doc_definition.items():


            self.sent_classifiers[k] = SentClassifiers( \
                                        input_dim = hidden_size,
                                        num_tags = 2,
                                        loss_reduction = self.loss_reduction,
                                        dropout = dropout_sent,
                                        sent_definition = sent_definition[k],
                                        )


            if self.concat_sent_scores:
                n = len(sent_definition[k])*2
            else:
                n = 0

            if self.project_sent:
                self.sent_ffnn[k] = FeedForward( \
                        input_dim = hidden_size+n,
                        num_layers = 1,
                        hidden_dims = self.project_size,
                        activations = get_activation('tanh'),
                        dropout = 0)

                out_dim = self.project_size
            else:
                out_dim = hidden_size + n


            self.sent_attention[k] = Attention( \
                                    input_dim = out_dim,
                                    dropout = dropout_doc,
                                    use_ffnn = True,
                                    activation = 'tanh',
                                    query_dim = attention_query_dim)

            self.doc_output_layers[k] = nn.Linear(out_dim, len(label_set))






        self.get_summary()


    def reset_parameters(self):

        i = 0
        parameter_count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.split('.')[0] != 'bert':
                    if name.split('.')[-1] == 'bias':
                        torch.nn.init.zeros_(param.data)
                    else:
                        torch.nn.init.normal_(param.data)
                    parameter_count += param.numel()

                i += 1
        logging.info(f"Reset {i} model variables")
        logging.info(f"Reset {parameter_count/1e6:.2f}M model parameters")

        logging.info(f"Reloaded BERT '{self.pretrained}'")
        self.bert = AutoModel.from_pretrained(self.pretrained)

        return True


    def forward(self, input_ids, attention_mask, verbose=False):

        # input_ids (document_count, sentence_count, sentence_length, embed_dim)
        # attention_mask (document_count, sentence_count, sentence_length)


        # Iterate over documents
        sent_vectors = []
        sent_mask = []
        sent_scores = OrderedDict()

        for in_ids, attn_mask in zip(input_ids, attention_mask):

            # in_ids (sentence_count, sequence_length)
            # attn_mask (sentence_count, sequence_length)

            # (sentence_count, embed_dim)
            sent_vec = self.bert(in_ids, attention_mask=attn_mask)['pooler_output']
            sent_vectors.append(sent_vec)

            # sentence scores
            for k in self.sent_classifiers:
                if k not in sent_scores:
                    sent_scores[k] = OrderedDict()

                sent_scores_dict = self.sent_classifiers[k](sent_vec)

                for j, vals in sent_scores_dict.items():
                    if j not in sent_scores[k]:
                        sent_scores[k][j] = []
                    sent_scores[k][j].append(vals)


            # (sentence_count)
            sm = torch.sum(attn_mask, dim=1) > 0
            sent_mask.append(sm)

        # (document_count, sentence_count, embedding_dimension)
        sent_vectors = torch.stack(sent_vectors, dim=0)

        # (document_count, sentence_count)
        sent_mask = torch.stack(sent_mask, dim=0)



        for k, ss in sent_scores.items():
            for j, vals in ss.items():
                # (document_count, sentence_count, 2)
                sent_scores[k][j] = torch.stack(vals)

        sent_scores_all = OrderedDict()
        for k, ss in sent_scores.items():
            # (doc_count, sent_count, 2*num sent labels)
            sent_scores_all[k] = torch.cat([vals for j, vals in ss.items()], dim=2)


        # iterate over output targets
        doc_vectors = OrderedDict([(k, None) for k in self.sent_attention])
        for k in self.sent_attention:

            if self.concat_sent_scores:
                sent_vectors_aug = torch.cat([sent_vectors, sent_scores_all[k]], dim=2)
            else:
                sent_vectors_aug = sent_vectors

            if self.project_sent:
                sent_vectors_aug = self.sent_ffnn[k](sent_vectors_aug)


            # doc_vec (document_count, embed_dim)
            # alphas (document_count, sentence_count)
            doc_vecs, alphas = self.sent_attention[k](sent_vectors_aug, sent_mask)
            doc_vectors[k] = doc_vecs

        doc_scores = OrderedDict([(k, []) for k in self.sent_attention])
        for k, doc_vecs in doc_vectors.items():
            # (document count, label count)
            doc_scores[k] = self.doc_output_layers[k](doc_vecs)

        return (doc_scores, sent_scores)


    # OVERRIDE
    def fit(self, X, y, device=None, path=None, shuffle=True):
        '''


        Parameters
        ----------

        X: documents as list of strings [doc [str]]
        y: labels as list of dictionarys

        '''



        logging.info('')
        logging.info('='*72)
        logging.info("Fit")
        logging.info('='*72)

        # Get/set device
        set_model_device(self, device)

        # Configure training mode
        self.train()

        # Create data set
        dataset = self.dataset_class( \
                                X = X,
                                y = y,
                                pretrained = self.pretrained,
                                device = device,
                                doc_definition = self.doc_definition,
                                sent_definition = self.sent_definition,
                                max_length = self.max_length,
                                max_sent_count = self.max_sent_count,
                                linebreak_bound = self.linebreak_bound,
                                keep_ws = self.keep_ws)

        # Create data loader
        dataloader = DataLoader(dataset,  \
                                shuffle = shuffle,
                                batch_size = self.batch_size)

        # Create optimizer
        '''
        https://github.com/huggingface/transformers/issues/657

        pretrained = model.bert.parameters()
        # Get names of pretrained parameters (including `bert.` prefix)
        pretrained_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]

        new_params= [v for k, v in model.named_parameters() if k not in pretrained_names]

        optimizer = AdamW(
            [{'params': pretrained}, {'params': new_params, 'lr': learning_rate * 10}],
            lr=learning_rate,
        )




        )


        '''

        if self.lr_ratio == 1:
            optimizer = AdamW(self.parameters(), lr = self.lr)
        else:
            pretrained = self.bert.parameters()
            pretrained_names = [f'bert.{k}' for (k, v) in self.bert.named_parameters()]
            new_params = [v for k, v in self.named_parameters() if k not in pretrained_names]
            optimizer = AdamW(
                [{'params': pretrained}, {'params': new_params, 'lr': self.lr*self.lr_ratio}],
                lr=self.lr)
        # define cross entropy
        #cross_entropy  = nn.NLLLoss(reduction=self.loss_reduction)

        # Create loss plotter
        plotter = PlotLoss(path=path)

        # Create prf aggregator
        prf_agg = PRFAggregator()

        # Loop on epochs
        pbar = tqdm(total=self.num_epochs)
        for j in range(self.num_epochs):

            loss_epoch = 0
            losses_epoch = OrderedDict()
            prf = []

            # Loop on mini-batches
            for i, (input_ids, attention_mask, doc_labels, sent_labels) in enumerate(dataloader):

                verbose = False #(i == 0) and (j == 0)

                # Reset gradients
                self.zero_grad()

                doc_scores, sent_scores = self(input_ids, attention_mask, verbose=verbose)

                loss_dict = OrderedDict()
                for k in doc_labels:
                    loss_dict[f"doc_{k[0:3]}"] = F.cross_entropy( \
                                                    input = doc_scores[k],
                                                    target = doc_labels[k],
                                                    reduction = self.loss_reduction)


                if self.use_sent_objective:
                    for k in doc_labels:
                        ls = []
                        for t in sent_labels[k]:
                            scores = sent_scores[k][t]
                            labels = sent_labels[k][t]

                            doc_count, sent_count, _ = tuple(scores.shape)

                            scores = scores.view(doc_count*sent_count, -1)
                            labels = labels.view(doc_count*sent_count)

                            l = F.cross_entropy( \
                                input = scores,
                                target = labels,
                                reduction = self.loss_reduction)
                            ls.append(l)
                        ls = aggregate(torch.stack(ls), self.loss_reduction)
                        loss_dict[f"sent_{k[0:3]}"] = ls

                loss = [v for k, v in loss_dict.items() if v is not None]
                loss = aggregate(torch.stack(loss), self.loss_reduction)


                plotter.update_batch(loss, loss_dict)

                #prf_agg.update_counts(self.perf_counts(y_true, y_pred))

                # Backprop loss
                loss.backward()

                loss_epoch += loss.item()
                for k, v in loss_dict.items():
                    if i == 0:
                        losses_epoch[k] = v.item()
                    else:
                        losses_epoch[k] += v.item()

                # Clip loss
                clip_grad_norm_(self.parameters(), self.grad_max_norm)

                # Update
                optimizer.step()

            plotter.update_epoch(loss_epoch, losses_epoch)

            msg = []
            msg.append('epoch={}'.format(j))
            msg.append('{}={:.1e}'.format('Total', loss_epoch))
            for k, ls in losses_epoch.items():
                msg.append('{}={:.1e}'.format(k, ls))

            #msg.append(prf_agg.prf())
            #prf_agg.reset()

            msg = ", ".join(msg)
            pbar.set_description(desc=msg)
            pbar.update()


        pbar.close()

        return True

    # OVERRIDE
    def predict(self, X, device=None, path=None):

        logging.info('')
        logging.info('='*72)
        logging.info("Predict")
        logging.info('='*72)

        # Get/set device
        set_model_device(self, device)

        # Configure training mode
        self.eval()

        # Create data set
        dataset = self.dataset_class( \
                                X = X,
                                pretrained = self.pretrained,
                                device = device,
                                doc_definition = self.doc_definition,
                                sent_definition = self.sent_definition,
                                max_length = self.max_length,
                                max_sent_count = self.max_sent_count,
                                linebreak_bound = self.linebreak_bound,
                                keep_ws = self.keep_ws)

        # Create data loader
        dataloader = DataLoader(dataset,  \
                                shuffle = False,
                                batch_size = self.batch_size)

        # deactivate autograd
        with torch.no_grad():

            pbar = tqdm(total=int(len(dataloader)/dataloader.batch_size))
            y = []
            for i, (input_ids, attention_mask) in enumerate(dataloader):

                verbose = False

                # Push data through model
                doc_scores, sent_scores = self(input_ids, attention_mask, verbose=verbose)


                y_batch = dataset.postprocess_y( \
                                    attention_mask  = attention_mask,
                                    doc_scores = doc_scores,
                                    sent_scores = sent_scores,
                                    )
                y.extend(y_batch)

                pbar.update()
            pbar.close()


        if path is not None:
            f = os.path.join(path, PREDICTIONS_FILE)
            joblib.dump(y, f)

        return y


    def perf_counts(self, y_true, y_pred):


        d = OrderedDict()
        if self.use_doc_classifier:
            doc_counts, sent_counts = self.doc_classifier.perf_counts( \
                            doc_labels = y_true["doc_labels"],
                            doc_scores = y_pred["doc_scores"],
                            sent_labels = y_true["sent_labels"],
                            sent_scores = y_pred["sent_scores"])
            d["doc"] = doc_counts
            d["sent"] = sent_counts



        return d
