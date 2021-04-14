
from collections import OrderedDict, Counter

from config.constants import SPAN_ONLY, NONE, TRIGGER



ATTRIBUTE_PATTERN = '{}Val'



ANATOMY = 'Anatomy'
ASSERTION = 'Assertion'
ASSERTION_VAL = 'AssertionVal'
CHANGE = 'Change'
CHANGE_VAL = 'ChangeVal'
CHARACTERISTICS = 'Characteristics'
DURATION = 'Duration'
FREQUENCY = 'Frequency'
SEVERITY = 'Severity'
SEVERITY_VAL = 'SeverityVal'

TEST_STATUS = 'TestStatus'
TEST_STATUS_VAL = 'TestStatusVal'

PRESENT = 'present'
ABSENT = 'absent'
POSSIBLE = 'possible'
CONDITIONAL = 'conditional'
HYPOTHETICAL = 'hypothetical'
NOT_PATIENT = 'not_patient'


POSITIVE = 'positive'
NEGATIVE = 'negative'
PENDING = 'pending'
CONDITIONAL = 'conditional'
NOT_ORDERED = 'not_ordered'
NOT_PATIENT = 'not_patient'
INDETERMINATE = 'indeterminate'

# Change
NO_CHANGE = 'no_change'
WORSENING = 'worsening'
IMPROVING = 'improving'
RESOLVED = 'resolved'

# Severity
MILD = 'mild'
MODERATE = 'moderate'
SEVERE = 'severe'

SSX = 'SSx'
COVID = 'COVID'

EVENT_TYPES = [SSX, COVID]
SPAN_ONLY_ARGUMENTS = [ANATOMY, CHARACTERISTICS, DURATION, FREQUENCY]


ENTITY_DEFINITION = OrderedDict()
ENTITY_DEFINITION[TRIGGER] =     [NONE, COVID, SSX]
ENTITY_DEFINITION[ASSERTION] =   [NONE, PRESENT, ABSENT, POSSIBLE, CONDITIONAL, HYPOTHETICAL, NOT_PATIENT]
ENTITY_DEFINITION[TEST_STATUS] = [NONE, POSITIVE, NEGATIVE, PENDING, CONDITIONAL, NOT_ORDERED, NOT_PATIENT, INDETERMINATE]
ENTITY_DEFINITION[CHANGE] =      [NONE, NO_CHANGE, WORSENING, IMPROVING, RESOLVED]
ENTITY_DEFINITION[SEVERITY] =    [NONE, MILD, MODERATE, SEVERE]
ENTITY_DEFINITION[SPAN_ONLY] =   [NONE, ANATOMY, CHARACTERISTICS, DURATION, FREQUENCY]

RELATION_DEFINITION = []
RELATION_DEFINITION.append((TRIGGER, ASSERTION))
RELATION_DEFINITION.append((TRIGGER, TEST_STATUS))
RELATION_DEFINITION.append((TRIGGER, CHANGE))
RELATION_DEFINITION.append((TRIGGER, SEVERITY))
RELATION_DEFINITION.append((TRIGGER, SPAN_ONLY))



norm_map = { \




    'guarding':             'abdominal',
    'rebound':              'abdominal',
    'fullness':             'abdominal',

    'confusion':            'altered mental status',
    'confused':             'altered mental status',
    'ams':                  'altered mental status',
    'delusions':            'altered mental status',
    'hallucinations':       'altered mental status',
    'delirium':             'altered mental status',
    'somnolent':            'altered mental status',
    'somnolence':           'altered mental status',
    'drowsy':               'altered mental status',
    'drowsiness':           'altered mental status',
    'sleepy':               'altered mental status',

    'anxious':              'anxiety',

    'agitated':             'agitation',


    'arthralgias':          'arthralgia',

    'blood':                'bleeding',
    'bloody':               'bleeding',
    'bleed':                'bleeding',

    'bluish':               'bluish',

    'blurry':               'blurred vision',
    'blurriness':           'blurred vision',
    'blurred':              'blurred vision',

    'bruise':               'bruising',
    'bruises':              'bruising',
    'ecchymosis':           'bruising',


    'rales':                'chest crackles',
    'rhonci':               'chest crackles',


    'cp':                   'chest pain',

    'chill':                'chills',
    'rigors':               'chills',

    'clubbing':             'clubbing',

    'constipated':          'constipation',

    'coughing':             'cough',
    'coughs':               'cough',
    'cough cough':          'cough',
    'distressed coughing':  'cough',
    'distress coughing':    'cough',
    'c':                    'cough',
    'c.':                   'cough',



    'cramps':               'cramping',

    'cyanosis':             'cyanosis',

    'dehydrated':           'dehydration',
    'dry':                  'dehydration',



    'poor po intake':       'decreased appetite',
    'poor p.o. intake':     'decreased appetite',
    'loss of appetite':     'decreased appetite',
    'reduced appetite':     'decreased appetite',
    'poor appetite':        'decreased appetite',
    'anorexia':             'decreased appetite',


    'deformity':            'deformities',


    'd':                    'diarrhea',
    'd.':                   'diarrhea',
    'loose stools':         'diarrhea',
    'diarrhea stools':      'diarrhea',

    'drainage':             'disharge',
    'oozing':               'disharge',
    'discharge':            'disharge',
    'exudate':              'disharge',
    'exudates':             'disharge',

    'distention':           'distended',

    'distressed':           'distress',


    'slurred speech':       'dysarthria',

    'difficulty swallowing':    'dysphagia',
    'dysphagia symptoms':       'dysphagia',
    'odynophagia':              'dysphagia',


    'erythematous':         'erythema',
    'redness':              'erythema',


    'falls':                'fall',

    'tired':                'fatigue',
    'fatigued':             'fatigue',
    'tiredness':            'fatigue',
    'lethargic':            'fatigue',
    'lethargy':             'fatigue',


    'fevers':               'fever',
    'febrile':              'fever',
    'f':                    'fever',
    'f.':                   'fever',

    'influenza - like symptoms': 'flu-like symptoms',
    'flu - like symptoms':       'flu-like symptoms',

    'abdominal symptoms':   'gi symptoms',

    'ha':                   'headache',
    'headaches':            'headache',
    'migraine':             'headache',

    'heart murmurs':         'heart murmur',

    'gerd symptoms':        'heartburn',
    'gerd symptom':         'heartburn',
    'heartburn symptoms':   'heartburn',
    'heartburn symptom':    'heartburn',


    'brbpr':                'hematochezia',
    'melena':               'hematochezia',

    'hematemesis':          'hematemesis',

    'hemoptysis':           'hemoptysis',

    'hernia':               'hernia',

    'incontinence':         'incontinent',

    'sick':                 'ill',
    'illness':              'ill',
    'ill - appearing':      'ill',
    'illness':              'ill',
    'ill appearing':        'ill',
    'ill symptoms':         'ill',

    'insomnia':             'insomnia',
    'insomniac':            'insomnia',


    'irritable':            'irritation',


    'icterus':              'jaundice',

    'stiffness':            'joint stiffness',
    'stiff':                'joint stiffness',

    'lightheaded':          'lightheadedness',
    'headedness':           'lightheadedness',
    'dizzy':                'lightheadedness',
    'dizziness':            'lightheadedness',

    'lymphadenopathy':      'lymphadenopathy',

    'malaise':              'malaise',

    'meningismus':          'meningismus',

    'ache':                 'myalgia',
    'aches':                'myalgia',
    'myalgias':             'myalgia',
    'bodyaches':            'myalgia',
    'aching':               'myalgia',


    'n':                    'nausea',
    'n.':                   'nausea',
    'nauseated':            'nausea',
    'nauseous':             'nausea',


    'tinnitus':             'neurologic symptoms',

    'organomegaly':         'organomegaly',

    'pains':                'pain',
    'painful':              'pain',
    'discomfort':           'pain',
    'aches':                'pain',
    'ache':                 'pain',
    'sore':                 'pain',
    'soreness':             'pain',
    'tender':               'pain',
    'tenderness':           'pain',





    'palpitation':          'palpitations',

    'tingling':             'paresthesia',
    'numbness':             'paresthesia',



    'pruritis':             'pruritus',
    'itch':                 'pruritus',
    'itches':               'pruritus',
    'itchy':                'pruritus',
    'itchiness':            'pruritis',


    'rashes':               'rash',
    'lesions':              'rash',
    'lesion':               'rash',
    'hives':                'rash',
    'hive':                 'rash',


    'respiratory symptom':  'respiratory symptoms',

    'rhinorrhea':           'runny nose',
    'uri symptoms':         'runny nose',
    'sneezing':             'runny nose',
    'congestion':           'runny nose',


    'seizure':              'seizures',

    'sob':                          'shortness of breath',
    'sob on exertion':              'shortness of breath',
    'dsypnea':                      'shortness of breath',
    'dyspnea':                      'shortness of breath',
    'dypsnea':                      'shortness of breath',
    'short of breath':              'shortness of breath',
    'out of breath':                'shortness of breath',
    'difficulty breathing':         'shortness of breath',
    'work of breathing':            'shortness of breath',
    'trouble breathing':            'shortness of breath',
    'respiratory distress':         'shortness of breath',
    'doe':                          'shortness of breath',
    'dyspnea on exertion':          'shortness of breath',
    'dyspnea exertion':             'shortness of breath',
    'shortneses of breath':         'shortness of breath',
    'difficult breathing':          'shortness of breath',
    'increase work of breathing':   'shortness of breath',
    'increased work of breathing':  'shortness of breath',
    'increased work of breathing':  'shortness of breath',
    'shortness breath':             'shortness of breath',
    'shortness of breaths':         'shortness of breath',
    'distress breathing':           'shortness of breath',
    'distressed breathing':         'shortness of breath',
    'difficulty of breathing':      'shortness of breath',
    '___shortness of breath':       'shortness of breath',

    'pharyngitis':          'sore throat',

    'sputum production':    'sputum',

    'sweating':             'sweats',
    'nightsweats':          'sweats',
    'diaphoresis':          'sweats',


    'edema':                'swelling',
    'oedema':               'swelling',
    'swollen':              'swelling',

    'fainting':             'syncope',
    'lightheaded':          'syncope',
    'pre-syncope':          'syncope',
    'pre - syncope':        'syncope',


    'thrush':               'thrush',

    'tremor':               'tremors',

    'ttp':                  'ttp',

    'ulcer':                'ulcers',
    'ulceration':           'ulcers',
    'ulcerations':          'ulcers',
    'wounds':               'ulcers',
    'wound':                'ulcers',
    'sores':                'ulcers',
    'pressure sores':       'ulcers',
    'pressure sore':        'ulcers',
    'abscess':              'ulcers',
    'abscesses':            'ulcers',

    'urinating':            'urination',


    'urinary':              'urinary symptoms',
    'dysuria':              'urinary symptoms',
    'hematuria':            'urinary symptoms',


    'v':                    'vomiting',
    'v.':                   'vomiting',
    'emesis':               'vomiting',
    'vomitting':            'vomiting',




    'wheeze':               'wheezing',
    'wheezes':              'wheezing',

    'weak':                 'weakness',




    'lossing weight':       'weight loss',
    'lost weight':          'weight loss',

    'gaining weight':       'weight gain',
    'gained weight':        'weight gain',
    'increased appetite':   'weight gain',

    }
