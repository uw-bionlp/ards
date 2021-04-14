from collections import OrderedDict, Counter
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import logging
import config.constants as constants
import copy

from corpus.utils import remove_white_space_at_ends

from config.constants import TRIGGER


class Entity(object):
    '''
    '''
    def __init__(self, type_, start, end, text, subtype=None, sent_index=None):
        self.type_ = type_
        self.start = start
        self.end = end
        self.text = text
        self.subtype = subtype
        self.sent_index = sent_index

    def indices(self):
        return (self.start, self.end)

    def __str__(self):
        x = ['{}={}'.format(k, v) for k, v in self.__dict__.items()]
        x = ', '.join(x)
        x = 'Entity({})'.format(x)
        return x

    def as_tuple(self):
        return tuple([v for k, v in self.__dict__.items()])


    def strip(self):

        self.text, self.start, self.end = \
                    remove_white_space_at_ends(self.text, self.start, self.end)

    def adjust_indices(self, offsets):
        '''
        adjust indices based on sentence offsets
        '''

        self.sent_index == None, "Sentence index is not None"

        self.strip()

        self.start, self.end, self.sent_index = \
                                adjust_indices(self.start, self.end, offsets)


    def get_key(self):
        return  (self.type_, self.subtype, self.start, self.end)

    def value(self):
        return  (self.type_, self.subtype, self.start, self.end, self.sent_index)

    def __eq__(self, other):
        return self.value() == other.value()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash('-'.join(self.value()))

class Relation(object):
    '''
    '''
    def __init__(self, entity_a, entity_b, role):
        self.entity_a = entity_a
        self.entity_b = entity_b
        self.role = role

    def __str__(self):
        x = ['{}={}'.format(k, v) for k, v in self.__dict__.items()]
        x = ', '.join(x)
        x = 'Relation({})'.format(x)
        return x

    def strip(self):

        self.entity_a.strip()
        self.entity_b.strip()

    def adjust_indices(self, offsets):
        '''
        adjust indices based on sentence offsets
        '''

        temp = copy.deepcopy(self)

        self.strip()

        self.entity_a.adjust_indices(offsets)
        self.entity_b.adjust_indices(offsets)

        if self.entity_a.sent_index != self.entity_b.sent_index:
            logging.warn("original: {}".format(temp))
            logging.warn('updated:  {}'.format(self))
            logging.warn("sent indices do not match: {} vs. {}".format(self.entity_a.sent_index, self.entity_b.sent_index))

class Event(object):
    '''
    '''
    def __init__(self, type_, arguments):
        self.type_ = type_

        assert isinstance(arguments, list)
        self.arguments = arguments

    def __str__(self):
        x = f'Event(type_={self.type_}, arguments=['
        for arg in self.arguments:
            x += '\n\t' + str(arg)
        x += '])'
        return x

def adjust_indices(start, end, offsets):
    '''
    adjust indices based on sentence offsets
    '''

    # iterate over sentence offsets
    sent_index = None
    for i, (sent_start, sent_end) in enumerate(offsets):

        # span start falls within current sentence
        if (start >= sent_start) and (start < sent_end):

            # issue warning if span end is outside of sentence
            if end > sent_end:
                logging.warn("span end not in same sentence as span start: ({}, {}, {})".format(start, end, offsets))

            # adjust indices
            start -= sent_start
            end -= sent_start
            sent_index = i


            # stop interacting
            break

    # issue warning if no match found
    if sent_index is None:
        ValueError("could not adjust indices: ({}, {}, {})".format(start, end, offsets))



    return (start, end, sent_index)

def tb2entities(tb_dict, attr_dict, attr_pat, as_dict=False):
    """
    convert textbound add attribute dictionaries to entities
    """



    # iterate over textbounds
    entities = OrderedDict()
    for tb_id, tb in tb_dict.items():

        # iterate over attributes
        subtype = None
        for attr_id, attr in attr_dict.items():

            # determine if textbound in type match
            tb_match = (attr.textbound == tb.id)
            type_match = (attr.attr == attr_pat.format(tb.type_))

            # update subtype if match
            if tb_match and type_match:
               subtype = attr.value
               break

        # create entity
        entity = Entity( \
            type_ = tb.type_,
            start = tb.start,
            end = tb.end,
            text = tb.text,
            subtype = subtype)

        assert tb_id not in entities
        entities[tb_id] = entity

    if as_dict:
        return entities
    else:
        return [entity for _, entity in entities.items()]

def tb2relations(relation_dict, tb_dict, attr_dict, attr_pat, as_dict=False):
    """
    convert textbound and relations to relation object
    """

    # get entities from textbounds
    entities = tb2entities(tb_dict, attr_dict, attr_pat=attr_pat, as_dict=True)

    # iterate over a relation dictionary
    relations = OrderedDict()
    for id, relation_brat in relation_dict.items():

        tb_1 = relation_brat.arg1
        tb_2 = relation_brat.arg2
        role = relation_brat.role

        relation = Relation( \
                entity_a = copy.deepcopy(entities[tb_1]),
                entity_b = copy.deepcopy(entities[tb_2]),
                role = role)

        assert id not in relations
        relations[id] = relation

    if as_dict:
        return relations
    else:
        return [relation for _, relation in relations.items()]


def brat2events(event_dict, tb_dict, attr_dict, attr_pat, as_dict=False):
    """
    convert textbound and relations to relation object
    """

    # get entities from textbounds
    entities = tb2entities(tb_dict, attr_dict, attr_pat=attr_pat, as_dict=True)

    # iterate over a relation dictionary
    events = OrderedDict()
    for id, event_brat in event_dict.items():

        # iterate over arguments
        arguments = []
        for i, (argument_type, tb_id) in enumerate(event_brat.arguments.items()):

            entity = entities[tb_id]

            # assume first entity is the trigger
            #if i == 0:
            #    entity.subtype = entity.type_
            #    entity.type_ = TRIGGER

            arguments.append(entity)

        event = Event( \
                type_ = event_brat.type_,
                arguments = arguments)

        assert id not in events
        events[id] = event

    if as_dict:
        return events
    else:
        return [event for _, event in events.items()]




def event2relations(event, role=1):

    trigger = event.arguments[0]
    assert trigger.type_ == TRIGGER

    relations = []
    for argument in event.arguments[1:]:
        relation = Relation(entity_a=trigger, entity_b=argument, role=role)
        relations.append(relation)

    return relations

def events2relations(events):

    relations = []
    for event in events:
        relations.extend(event2relations(event))

    return relations


def relations2events(relations):

    triggers = OrderedDict()
    arguments = OrderedDict()
    for relation in relations:

        entity_a = relation.entity_a
        entity_b = relation.entity_b


        key_a = entity_a.get_key()
        key_b = entity_b.get_key()



        if trig_key not in triggers:
            triggers[trig_key] = entity_a

        if trig_key not in arguments:
            arguments[trig_key]
