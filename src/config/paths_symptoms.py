



import os



"""
Input paths
"""

symptoms_brat = '/home/data/bionlp/Covid19_SignSymptom/'
symptoms_doc_tags = '/home/lybarger/clinical_extractors/resources/covid_tags_by_doc.csv'
symptoms_original = '/home/lybarger/clinical_extractors/resources/orig_covid.json'


"""
Output paths
"""

analyses = '/home/lybarger/clinical_extractors/analyses_symptoms/'


brat_import = os.path.join(analyses, "step010_brat_import")
modeling = os.path.join(analyses,    "step420_symptom_modeling")
