
from collections import Counter
from collections import OrderedDict


from config.constants import EVENT, RELATION, TEXTBOUND, ATTRIBUTE, ATTRIBUTE_PATTERN
from corpus.document import Document
from corpus.brat import get_annotations, write_txt, write_ann
from corpus.labels import tb2entities, tb2relations, adjust_indices, brat2events
from corpus.align import align_whitespace, get_matching_text
from corpus.utils import remove_white_space_at_ends

def non_white_space_check(x, y):
    x = "".join(x.split())
    y = "".join(y.split())
    assert x == y, '''"{}" vs "{}"'''.format(x, y)




class DocumentBrat(Document):


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


        Document.__init__(self, \
            id = id,
            text_ = text_,
            tags = tags,
            patient = patient,
            date = date,
            tokenizer = tokenizer
            )


        self.ann = ann
        self.attr_pat = attr_pat


        # Extract events, text bounds, and attributes from annotation string
        self.event_dict, self.relation_dict, self.tb_dict, self.attr_dict = get_annotations(ann)




    def entities(self, adjust_offsets=False):
        '''
        get list of entities for document
        '''

        entities = tb2entities(self.tb_dict, self.attr_dict, attr_pat=self.attr_pat)

        # rearrange entities by sentence
        if adjust_offsets:

            # iterate over entities
            sents = self.sents()
            for entity in entities:
                assert entity.start != None, f'{self.id} --- {entity}'
                entity.adjust_indices(self.sent_offsets())

                a = entity.text
                b = sents[entity.sent_index][entity.start:entity.end]

                assert a == b, '"{}: {}" VS. "{}"'.format(self.id, a, b)

        return entities



    def relations(self, adjust_offsets=False):
        '''
        get list of relations for document
        '''
        relations = tb2relations(self.relation_dict, self.tb_dict, self.attr_dict, attr_pat=self.attr_pat)

        # rearrange relations bi sentence
        if adjust_offsets:

            # iterate over relations
            sents = self.sents()
            for relation in relations:
                relation.adjust_indices(self.sent_offsets())

        return relations

    def events(self):
        '''
        get list of entities for document
        '''

        return brat2events(self.event_dict, self.tb_dict, self.attr_dict, attr_pat=self.attr_pat)


    def labels(self, adjust_offsets=False):



        events = self.events()

        relations = self.relations(adjust_offsets=adjust_offsets)

        entities = self.entities(adjust_offsets=adjust_offsets)



        return (events, relations, entities)

    # OVERRIDE
    def update_text(self, text, tokenizer, char_map):

        self.spacy_obj = tokenizer(text)

        #print()
        #print('='*72)
        #print(text)
        #print('='*72)


        for id, tb in self.tb_dict.items():
            #print()
            #print(tb.start, tb.end, tb.text)

            assert tb.start in char_map, '{} not {}'.format(tb.start, char_map)
            assert tb.end in char_map, '{} not {}'.format(tb.end, char_map)

            tb.start = char_map[tb.start]
            tb.end = char_map[tb.end]
            tb.text = text[tb.start:tb.end]
            #print(tb.start, tb.end, tb.text)

        #print()

    def update_text_whitespace(self, text, tokenizer, find_matching=True):


        text_original = self.text()
        text_new = text

        text = None


        if find_matching:
            text_new = get_matching_text(text_original, text_new)
            non_white_space_check(text_original, text_new)

        # to get map from current taxed to new text
        char_map = align_whitespace(text_original, text_new)

        # update text
        self.spacy_obj = tokenizer(text_new)

        # update textbound character indices
        for id, tb in self.tb_dict.items():

            start_original = tb.start
            end_original = tb.end

            start_new = char_map[start_original]
            end_new = char_map[end_original]

            text_tb = text_new[start_new:end_new]

            # check text substitution
            non_white_space_check(tb.text, text_tb)

            # remove trailing and leading white
            text_tb, start_new, end_new = \
                        remove_white_space_at_ends(text_tb, start_new, end_new)

            assert text_tb == text_new[start_new:end_new]
            non_white_space_check(tb.text, text_tb)

            # update indices
            tb.start = start_new
            tb.end = end_new
            tb.text = text_tb

    def brat_str(self):

        ann = []
        for _, x in self.tb_dict.items():
            ann.append(x.brat_str())
        for _, x in self.relation_dict.items():
            ValueError('NEED TO COMPLETE')
            #ann.append(x.brat_str())
        for _, x in self.event_dict.items():
            ann.append(x.brat_str())
        for _, x in self.attr_dict.items():
            ann.append(x.brat_str())
        ann = "\n".join(ann)

        return ann

    def write_brat(self, path):

        fn_text = write_txt(path, self.id, self.text())
        fn_ann = write_ann(path, self.id, self.brat_str())

        return (fn_text, fn_ann)

    def quality_check(self):
        return (self.id, "no checks defined")


    def annotation_summary(self):

        counter = Counter()
        counter[EVENT] += len(self.event_dict)
        counter[RELATION] += len(self.relation_dict)
        counter[TEXTBOUND] += len(self.tb_dict)
        counter[ATTRIBUTE] += len(self.attr_dict)

        return counter

    def label_summary(self):

        counter = Counter()
        counter[EVENT] += len(self.event_dict)
        counter[RELATION] += len(self.relation_dict)
        counter[TEXTBOUND] += len(self.tb_dict)
        counter[ATTRIBUTE] += len(self.attr_dict)

        return counter


    def snap_textbounds(self):
        '''
        Snap the textbound indices to the starts and ends of the associated tokens.
        This is intended to correct annotation errors where only a partial word is annotated.
        '''
        offsets = self.token_offsets()
                #print(type(token))
                #print(token.text, token.offset)


        text = self.text()

        for id, tb in self.tb_dict.items():


            _, start_tb, end_tb = \
                        remove_white_space_at_ends(tb.text, tb.start, tb.end)


            # Adjust start
            start_new = None
            for _, start, end in offsets:
                if (start_tb >= start) and (start_tb < end):
                    start_new = start
                    break

            # Adjust end
            end_new = None
            for _, start, end in offsets:
                if (end_tb > start) and (end_tb <= end):
                    end_new = end
                    break

            if (start_new is None) or (end_new is None):
                ValueError(f"Could not map textbound:\n{tb}\n{text}")


            #if 'ilar atele' in tb.text:
            #    print("FOUND 'ilar atele'")

            if (tb.start != start_new) or (tb.end != end_new):
                #print()
                #print(tb)
                text_new = text[start_new:end_new]

                tb.start = start_new
                tb.end = end_new
                tb.text = text_new
                #print(tb)
                #print(self.tb_dict[id])

                assert self.tb_dict[id].text == text_new


        return True
