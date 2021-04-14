from collections import Counter, OrderedDict

from scoring.scorer import Scorer
from scoring.scoring_utils import compare_entities, compare_relations
from scoring.scoring_utils import compare_doc_labels
from config.constants import ENTITIES, RELATIONS, DOC_LABELS
from utils.misc import list_to_dict

class ScorerSymptoms(Scorer):


    def __init__(self):


        Scorer.__init__(self)

        self.exact = False


    # OVERRIDE
    def compare(self, T, P):


        NotImplementedError("need to implement scorer")
        
        assert len(T) == len(P)

        T = list_to_dict(T)
        P = list_to_dict(P)

        assert len(T) == len(P)

        dfs = OrderedDict()

        df = compare_doc_labels(T[DOC_LABELS], P[DOC_LABELS])
        dfs[DOC_LABELS] = df

        df = compare_entities(T[ENTITIES], P[ENTITIES], exact=self.exact, subtype=True)
        dfs[ENTITIES] = df

        df = compare_entities(T[ENTITIES], P[ENTITIES], exact=self.exact, subtype=False)
        dfs["entities_no_subtyp_match"] = df

        df = compare_relations(T[RELATIONS], P[RELATIONS], exact=self.exact)
        dfs[RELATIONS] = df


        return dfs
