



import os



"""
Input paths
"""

sdoh_brat = '/home/data/bionlp/social_determinants/'
sdoh_brat_partial = '/home/data/bionlp/social_determinants_partial'
symptoms_brat = '/home/data/bionlp/Covid19_SignSymptom/'

sdoh_doc_tags = '/home/lybarger/clinical_extractors/resources/sdoh_tags_by_doc.csv'
symptoms_doc_tags = '/home/lybarger/clinical_extractors/resources/covid_tags_by_doc.csv'

sdoh_original = "/home/lybarger/clinical_extractors/resources/orig_sdoh.json"
symptoms_original = "/home/lybarger/clinical_extractors/resources/orig_covid.json"

sdoh_brat_deid = '/home/data/bionlp/deid_social_det/'




"""
Output paths
"""

analyses = '/home/lybarger/clinical_extractors/analyses_deid/'


text_import    = os.path.join(analyses, "step005_text_import")

brat_import    = os.path.join(analyses, "step010_brat_import")
# brat_adjust      = os.path.join(analyses, "step012_brat_adjust")
brat_deid_auto   = os.path.join(analyses, "step014_brat_deid_auto")
brat_deid_anno   = os.path.join(analyses, "step016_brat_deid_anno")
brat_deid_final  = os.path.join(analyses, "step018_brat_deid_final")
