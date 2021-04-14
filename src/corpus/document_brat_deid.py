from copy import deepcopy
import os
import copy

from collections import Counter
from corpus.document_brat import DocumentBrat
from config.constants_deid import DEID_FOOTER, REVIEW_STATUS, REVIEW_STATUS_
from config.constants_deid import PENDING, DONE



from scoring.scoring_utils import entity_hist, relation_hist



class DocumentBratDeid(DocumentBrat):


    def __init__(self, \
        id,
        text_,
        ann,
        tags = None,
        patient = None,
        date = None,
        tokenizer = None,
        ):


        DocumentBrat.__init__(self, \
            id = id,
            text_ = text_,
            ann = ann,
            tags = tags,
            patient = patient,
            date = date,
            tokenizer = tokenizer
            )


    # OVERRIDE
    def quality_check(self):

        id = self.id
        annotator, subset, source, index = id.split(os.sep)

        error_log = []

        # make sure review status is DONE
        review_status = [entity for id, entity in self.entities().items() \
                                        if entity.type_ == REVIEW_STATUS]
        assert len(review_status) == 1
        review_status = review_status[0]
        if review_status.subtype == DONE:
            pass
        else:
            msg = "NOT DONE"
            error_log.append((id, annotator, msg))

        return error_log


    # OVERRIDE
    def labels(self):

        assert len(self.event_dict) == 0
        assert len(self.relation_dict) == 0

        entities = self.entities()
        entities = [entity for id, entity in entities.items() \
                                if entity.type_ != REVIEW_STATUS]
        return entities


    # OVERRIDE
    def label_summary(self):

        counter_subtype = Counter()
        counter_text = Counter()

        entities = self.labels()
        for entity in entities:
            counter_subtype[(entity.subtype)] += 1
            counter_text[(entity.subtype, entity.text)] += 1

        return (counter_subtype, counter_text)
