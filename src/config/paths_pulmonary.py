
import os



"""
Input paths
"""

pna_cpis = '/data/common/chestxrayPNACPIS/'
cos = '/data/common/xray-cos/events-COS-corpus/release_Jan_1_2015/data/'
covid_xray_notes =  '/home/data/bionlp/COVID_ChestXray/Radiographdata.xlsx'
covid_xray_labels = "/home/data/bionlp/COVID_ChestXray/Copy of CHROme_cxr_Pipavath_adjudication_edm_Day 0_Day 1 Pipavath reads.xlsx"

xray_quadrant_interp = '/home/data/bionlp/COVID_ChestXray/Pipavath_reads_KJL_consolidation.csv'

emerge_xray_notes = '/home/data/bionlp/eMERGE/emerge_note_data.json'
brat_xray = '/home/data/bionlp/ARDS'
pulmonary_doc_tags = '/home/lybarger/clinical_extractors/resources/pulmonary_subset_assignments.csv'



"""
Output paths
"""

analyses = '/home/lybarger/clinical_extractors/analyses_pulmonary/'

text_import = os.path.join(analyses, "step005_text_import")
compare_docs = os.path.join(analyses, "step007_compare_docs")
brat_import = os.path.join(analyses, "step010_brat_import")
agreement   = os.path.join(analyses, "step012_annotator_agreement")

modeling = os.path.join(analyses,   "step320_pulmonary_modeling")
summary = os.path.join(analyses,   "step321_pulmonary_summary")
discrete = os.path.join(analyses,   "step322_pulmonary_discrete")
stats = os.path.join(analyses, "step324_pulmonary_corpus_stats")
predict = os.path.join(analyses,   "step325_pulmonary_predict")
image_anno_comp = os.path.join(analyses, "step326_image_anno_compare")
