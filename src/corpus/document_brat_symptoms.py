from copy import deepcopy
import os
import copy

from collections import Counter, OrderedDict
from corpus.document_brat import DocumentBrat
from scoring.scoring_utils import entity_hist, relation_hist
from config.constants import ENTITIES, RELATIONS, DOC_LABELS, EVENTS, TRIGGER, SPAN_ONLY
from config.constants_symptoms import ATTRIBUTE_PATTERN, SPAN_ONLY_ARGUMENTS, EVENT_TYPES



def event_map(event):


    for i, entity in enumerate(event.arguments):

        if i == 0:
            assert entity.type_ in EVENT_TYPES
            trigger_start = entity.start
            trigger_end = entity.end


        # entity is trigger
        if entity.type_ in EVENT_TYPES:
            entity.subtype = entity.type_
            entity.type_ = TRIGGER


        # entity is a span only argument
        elif entity.type_ in SPAN_ONLY_ARGUMENTS:
            assert entity.subtype is None
            entity.subtype = entity.type_
            entity.type_ = SPAN_ONLY

        # entity is a labeled argument and not a trigger
        else:
            assert entity.subtype is not None, entity
            entity.start = trigger_start
            entity.end = trigger_end






class DocumentBratSymptoms(DocumentBrat):


    def __init__(self, \
        id,
        text_,
        ann,
        tags = None,
        patient = None,
        date = None,
        tokenizer = None,
        attr_pat = ATTRIBUTE_PATTERN
        ):


        DocumentBrat.__init__(self, \
            id = id,
            text_ = text_,
            ann = ann,
            tags = tags,
            patient = patient,
            date = date,
            tokenizer = tokenizer,
            attr_pat = attr_pat
            )


    # OVERRIDE
    def labels(self):


        assert len(self.relation_dict) == 0

        events = self.events()

        return events




    #
    def Xy(self, swap_index=True):

        X = self.text()

        events = self.labels()


        for event in events:
            event_map(event)

        entities = [entity for event in events for entity in event.arguments]


        y = OrderedDict()
        y[ENTITIES] = entities
        y[EVENTS] = events

        return (X, y)
