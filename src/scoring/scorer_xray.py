from collections import Counter, OrderedDict
import pandas as pd


from scoring.scorer import Scorer
from scoring.scoring_utils import compare_entities, compare_relations
from scoring.scoring_utils import compare_doc_labels, compare_sent_labels
from config.constants import ENTITIES, RELATIONS, DOC_LABELS, SENT_LABELS
from config.constants import DOC_LABELS_SUMMARY, SENT_LABELS_SUMMARY, TYPE, SUBTYPE, ARG_1, ARG_2, ROLE, SUBTYPE_A, SUBTYPE_B, RELATIONS_SUMMARY, ENTITIES_SUMMARY
from utils.misc import nest_list
from scoring.scoring_utils import PRF

NO_SUB = "entities_no_subtyp_match"
PARTIAL = "entities_partial"
PARTIAL_SUMMARY = "entities_partial_summary"

class ScorerXray(Scorer):


    def __init__(self):


        Scorer.__init__(self)

        self.match_type = "overlap"



    # OVERRIDE
    def compare(self, T, P):



        assert len(T) == len(P), f"{len(T)} vs {len(P)}"




        T = nest_list(T)
        P = nest_list(P)

        #assert len(T) == len(P)

        dfs = OrderedDict()

        df = compare_doc_labels(T[DOC_LABELS], P[DOC_LABELS])
        dfs[DOC_LABELS] = df

        df = PRF(df.groupby(TYPE).sum())
        dfs[DOC_LABELS_SUMMARY] = df



        if SENT_LABELS in P:
            df = compare_sent_labels(T[SENT_LABELS] , P[SENT_LABELS])
            dfs[SENT_LABELS] = df

            df = PRF(df.groupby(DOC_LABELS).sum())
            dfs[SENT_LABELS_SUMMARY] = df

        if ENTITIES in P:
            df = compare_entities(T[ENTITIES], P[ENTITIES], \
                                            match_type = self.match_type,
                                            subtype = True)
            dfs[ENTITIES] = df

            df = PRF(df.groupby(TYPE).sum())
            dfs[ENTITIES_SUMMARY] = df


            df = compare_entities(T[ENTITIES], P[ENTITIES],
                                            match_type = self.match_type,
                                            subtype = False)
            dfs[NO_SUB] = df


            df = compare_entities(T[ENTITIES], P[ENTITIES], \
                                            match_type = "partial",
                                            subtype = True)
            dfs[PARTIAL] = df

            df = PRF(df.groupby(TYPE).sum())
            dfs[PARTIAL_SUMMARY] = df


        if RELATIONS in P:
            df = compare_relations(T[RELATIONS], P[RELATIONS], \
                                            match_type=self.match_type)
            dfs[RELATIONS] = df

            df = PRF(df.groupby(ARG_1).sum())
            dfs[RELATIONS_SUMMARY] = df


        return dfs

    # OVERRIDE
    def combine(self, dfs):

        dfs_out = OrderedDict()

        df = pd.concat(dfs[DOC_LABELS], axis=0).groupby([TYPE, SUBTYPE]).mean()
        dfs_out[DOC_LABELS] = df

        df = PRF(df.groupby(TYPE).sum())
        dfs_out[DOC_LABELS_SUMMARY] = df


        if SENT_LABELS in dfs:
            df = pd.concat(dfs[SENT_LABELS], axis=0).groupby([DOC_LABELS, SUBTYPE_A, SUBTYPE_B]).mean()
            dfs_out[SENT_LABELS] = df

            df = PRF(df.groupby(DOC_LABELS).sum())
            dfs_out[SENT_LABELS_SUMMARY] = df


        if ENTITIES in dfs:
            dfs_out[ENTITIES] =   pd.concat(dfs[ENTITIES],   axis=0).groupby([TYPE, SUBTYPE]).mean()

        if NO_SUB in dfs:
            dfs_out[NO_SUB]  =    pd.concat(dfs[NO_SUB],     axis=0).groupby([TYPE]).mean()

        if RELATIONS in dfs:
            dfs_out[RELATIONS]  = pd.concat(dfs[RELATIONS],  axis=0).groupby([ARG_1, ARG_2, ROLE]).mean()

        return dfs_out
