from copy import deepcopy
import os
import copy

from collections import Counter, OrderedDict
from corpus.document_brat import DocumentBrat
from config.constants_pulmonary import ASSERTION, EXTRAPARENCHYMAL
from config.constants_pulmonary import LABELS, INFILTRATES_FOOTER, EXTRAPARENCHYMAL_FOOTER
from config.constants_pulmonary import SIZE, SIDE, NEGATION, REGION, NONE, OTHER
from scoring.scoring_utils import entity_hist, relation_hist
from config.constants_pulmonary import FOOTER, INFILTRATES, EXTRAPARENCHYMAL
from config.constants import ENTITIES, RELATIONS, DOC_LABELS, SENT_LABELS
from config.constants_pulmonary import ATTRIBUTE_PATTERN
from corpus.labels import Entity, Relation
from layers.xfmr2 import tokenize_documents
from layers.text_class_utils import get_sent_labels_multi

def assertion_map(entity):

    if entity.type_ == ASSERTION:
        if INFILTRATES_FOOTER in entity.text:
            entity.type_ = INFILTRATES_FOOTER.lower()
        elif EXTRAPARENCHYMAL_FOOTER in entity.text:
            entity.type_ = EXTRAPARENCHYMAL_FOOTER.lower()
        #else:
        #    ValueError("cannot resolve type:\t{}".format(entity))
    return entity


def negation_map(entity):
    if entity.type_ == NEGATION:
        entity.subtype = NEGATION

    return entity



def swap_sidedness(entities, relations, role=1):


    region_entities = []
    for entity in entities:
        if (entity.type_ == REGION):
            region_entities.append(entity)


    new_relations = []
    for region in region_entities:

        found = []

        # iterate over relations
        for relation in relations:

            entity_a = relation.entity_a
            entity_b = relation.entity_b


            assert entity_a.type_ == REGION
            assert entity_b.type_ != REGION


            region_match = (region.get_key() == entity_a.get_key())
            type_match = (entity_b.type_ != SIZE)

            if region_match and type_match:

                if entity_b.type_ == SIDE:
                    subtype = entity_b.subtype
                elif entity_b.type_ == NEGATION:
                    subtype = NEGATION
                else:
                    raise ValueError("invalid type")

                side = Entity( \
                        type_ = SIDE,
                        start = region.start,
                        end = region.end,
                        text = region.text,
                        subtype = subtype,
                        sent_index = None)

                if side.value() not in found:
                    found.append(side.value())

                    R = Relation( \
                            entity_a = region,
                            entity_b = side,
                            role = role)
                    new_relations.append(R)



        if len(found) == 0:

            side = Entity( \
                    type_ = SIDE,
                    start = region.start,
                    end = region.end,
                    text = region.text,
                    subtype = OTHER,
                    sent_index = None)

            R = Relation( \
                    entity_a = region,
                    entity_b = side,
                    role = role)

            new_relations.append(R)

    #for relation in relations:
    #    print(relation)

    #print()
    #for relation in new_relations:
    #    print(relation)


    new_relations = sorted(new_relations, key=lambda x: (x.entity_a.start, x.entity_b.start))

    new_entities = []
    found = []
    for relation in new_relations:
        entity_a = relation.entity_a
        entity_b = relation.entity_b

        if entity_a.value() not in found:
            new_entities.append(entity_a)
            found.append(entity_a.value())

        if entity_b.value() not in found:
            new_entities.append(entity_b)
            found.append(entity_b.value())


    new_entities = sorted(new_entities, key=lambda x: x.start)
    #for entity in new_entities:
    #    print(entity)
    #z = sldkj

    return (new_entities, new_relations)

class DocumentBratXray(DocumentBrat):


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
    def quality_check(self):


        error_log = []

        id = self.id
        id_split = id.split(os.sep)
        annotator = id_split[1]

        # make sure entity map can be applied
        for entity in self.entities():
            if entity.type_ == ASSERTION:
                if (INFILTRATES_FOOTER in entity.text) or \
                    (EXTRAPARENCHYMAL_FOOTER in entity.text):
                    pass
                else:
                    msg = "invalid assertion selection for text '{}'".format(entity.text)
                    error_log.append((id, annotator, msg))

        # make sure all arguments with subtype have defined subtype
        for entity in self.entities():

            # entity type requires subtype
            if (entity.type_ in LABELS):

                allowable = LABELS[entity.type_]

                # Invalid or undefined subtype
                if (allowable is not None) and (entity.subtype not in allowable):

                    if (entity.type_ == ASSERTION) and (entity.subtype is None):
                        msg = "'{}' label for '{}' missing subtype value from {}".format(entity.type_, entity.text, allowable)
                    else:
                        msg = "'{}' assigned value of '{}'. must be in {}".format(entity.type_, entity.subtype, allowable)
                    error_log.append((id, annotator, msg))

        # to make sure all nodes have ASSERTION labels at end
        entities = copy.deepcopy(self.entities())
        types = []
        for v in entities:
            assertion_map(v)
            types.append(v.type_)
        for type_ in [INFILTRATES_FOOTER, EXTRAPARENCHYMAL_FOOTER]:
            if type_.lower() not in types:
                msg = '{} not labeled at the end of note'.format(type_)
                error_log.append((id, annotator, msg))

        # make sure a specific entities are connected to region
        entities_in_relations = []
        for relation in self.relations():
            entities_in_relations.append(relation.entity_a.as_tuple())
            entities_in_relations.append(relation.entity_b.as_tuple())
        entities = [entity.as_tuple() for entity in entities]
        for entity in entities:
            if (entity[0] in [SIDE, SIZE, REGION, NEGATION]) and \
               (entity not in entities_in_relations):

                msg = "'{}' not connected through attr to another entity.".format(entity[0])
                error_log.append((id, annotator, msg))




        return error_log


    # OVERRIDE
    def labels(self, adjust_offsets=False):

        assert len(self.event_dict) == 0

        entities = self.entities(adjust_offsets=adjust_offsets)
        for entity in entities:
            assertion_map(entity)
            negation_map(entity)

        relations = self.relations(adjust_offsets=adjust_offsets)

        for relation in relations:
            relation.entity_a = negation_map(relation.entity_a)
            relation.entity_b = negation_map(relation.entity_b)


        return (entities, relations)

    # OVERRIDE
    def label_summary(self):

        entities, relations = self.labels()

        counter_entities_sub = entity_hist(entities, subtype=True)
        counter_entities_no_sub = entity_hist(entities, subtype=False)
        counter_relations = relation_hist(relations)

        return (counter_entities_sub, counter_entities_no_sub, counter_relations)


    def X(self):

        X = self.text()
        if FOOTER in X:
            X = X.replace(FOOTER, '')

        return X

    def y(self, doc_map=None, side_swap=False):

        y = OrderedDict()
        entities_all, relations = self.labels()

        #for entity in entities_all:
        #    sent = X[entity.sent_index]
        #    span = sent[entity.start:entity.end]
        #    assert entity.text == span, f"{repr(entity.text)} vs {repr(span)}"

        infiltrates = None
        extraparenchymal = None
        entities = []
        for entity in entities_all:
            if entity.type_ == INFILTRATES:
                infiltrates = entity.subtype
            elif entity.type_ == EXTRAPARENCHYMAL:
                extraparenchymal = entity.subtype
            else:
                entities.append(entity)

        assert infiltrates is not None
        assert extraparenchymal is not None


        if doc_map is not None:
            infiltrates = doc_map[infiltrates]
            extraparenchymal = doc_map[extraparenchymal]
        y[DOC_LABELS] = OrderedDict([(INFILTRATES, infiltrates), \
                                     (EXTRAPARENCHYMAL, extraparenchymal)])


        if side_swap:
            entities, relations = swap_sidedness(entities, relations)


        y[ENTITIES] = entities
        y[RELATIONS] = relations



        return y


    # OVERRIDE
    def Xy(self, doc_map=None, side_swap=False):

        X = self.X()

        y = self.y(doc_map=doc_map, side_swap=side_swap)

        return (X, y)

    '''
    def Xy(self, doc_map=None):

        #print("="*100)
        #for sent, (a, b) in zip(self.sents(), self.sent_offsets()):
        #    print(a, b, '"{}"'.format(sent))
        #    print('-'*20)
        #print("="*100)

        X = self.sents()
        sent_offsets = self.sent_offsets()
        assert len(X) == len(sent_offsets)

        y = OrderedDict()

        entities_all, relations = self.labels(adjust_offsets=True)




        for entity in entities_all:
            sent = X[entity.sent_index]
            span = sent[entity.start:entity.end]
            assert entity.text == span, f"{repr(entity.text)} vs {repr(span)}"


        infiltrates = None
        extraparenchymal = None
        entities = []
        for entity in entities_all:
            if entity.type_ == INFILTRATES:
                infiltrates = entity.subtype
            elif entity.type_ == EXTRAPARENCHYMAL:
                extraparenchymal = entity.subtype
            else:
                entities.append(entity)

        assert infiltrates is not None
        assert extraparenchymal is not None


        if doc_map is not None:
            infiltrates = doc_map[infiltrates]
            extraparenchymal = doc_map[extraparenchymal]
        y[DOC_LABELS] = OrderedDict([(INFILTRATES, infiltrates), \
                                     (EXTRAPARENCHYMAL, extraparenchymal)])


        y[ENTITIES] = entities
        y[RELATIONS] = relations


        removed = X[-4:]
        X = X[:-4]
        removed = ''.join(''.join(removed).split())
        footer = ''.join(FOOTER.split())
        assert removed == footer


        return (X, y)
    '''
