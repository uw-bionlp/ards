

'''
Determinants
'''
ALCOHOL = 'Alcohol'
COUNTRY = 'Country'
DRUG = 'Drug'
EMPLOYMENT = 'Employment'
ENVIRO_EXPOSURE = 'EnviroExposure'
GENDER_ID = 'GenderID'
INSURANCE = 'Insurance'
LIVING_STATUS = 'LivingStatus'
PHYS_ACTIVITY = 'PhysActivity'
RACE = 'Race'
SEXUAL_ORIENT = 'SexualOrient'
TOBACCO = 'Tobacco'
OCCUPATION = 'Occupation'
ENVIRONMENTAL_EXPOSURE = 'EnvironmentalExposure'
LIVING_SIT = 'LivingSituation'
PHYSICAL_ACTIVITY = 'PhysicalActivity'

'''
Entities
'''

TRIGGER = 'Trigger'

# Span and class - new
STATUS_TIME             = 'StatusTime'
STATUS_TIME_VAL         = 'StatusTimeVal'

DEGREE                  = 'Degree'
DEGREE_VAL              = 'DegreeVal'

STATUS_EMPLOY           = 'StatusEmploy'
STATUS_EMPLOY_VAL       = 'StatusEmployVal'

STATUS_INSURE           = 'StatusInsure'
STATUS_INSURE_VAL       = 'StatusInsureVal'

TYPE_GENDER_ID          = 'TypeGenderID'
TYPE_GENDER_ID_VAL      = 'TypeGenderIDVal'

TYPE_LIVING             = 'TypeLiving'
TYPE_LIVING_VAL         = 'TypeLivingVal'

TYPE_SEXUAL_ORIENT      = 'TypeSexualOrient'
TYPE_SEXUAL_ORIENT_VAL  = 'TypeSexualOrientVal'

# Span and class - previous
STATUS = 'Status'
STATE = 'State'

# Span only - new
AMOUNT      = 'Amount'
DURATION    = 'Duration'
FREQUENCY   = 'Frequency'
HISTORY     = 'History'
METHOD      = 'Method'
TYPE        = 'Type'

# Span only - previous
EXPOSURE_HISTORY = 'ExposureHistory'
QUIT_HISTORY = 'QuitHistory'
LOCATION = 'Location'


export_map = {}

export_map['arg_map'] = {}
export_map['arg_map'][ALCOHOL] = {}
export_map['arg_map'][DRUG] = {}
export_map['arg_map'][TOBACCO] = {}
export_map['arg_map'][EMPLOYMENT] = {}
export_map['arg_map'][ENVIRO_EXPOSURE] = {}
export_map['arg_map'][GENDER_ID] = {}
export_map['arg_map'][INSURANCE] = {}
export_map['arg_map'][LIVING_STATUS] = {}
export_map['arg_map'][PHYS_ACTIVITY] = {}
export_map['arg_map'][SEXUAL_ORIENT] = {}

export_map['tb_map'] = {}
export_map['tb_map'][ALCOHOL] =         {STATUS: STATUS_TIME}
export_map['tb_map'][DRUG] =            {STATUS: STATUS_TIME}
export_map['tb_map'][TOBACCO] =         {STATUS: STATUS_TIME}
export_map['tb_map'][EMPLOYMENT] =      {STATUS: STATUS_EMPLOY}
export_map['tb_map'][ENVIRO_EXPOSURE] = {STATUS: STATUS_TIME}
export_map['tb_map'][GENDER_ID] =       {STATUS: STATUS_TIME, TYPE: TYPE_GENDER_ID}
export_map['tb_map'][INSURANCE] =       {STATUS: STATUS_INSURE}
export_map['tb_map'][LIVING_STATUS] =   {STATUS: STATUS_TIME, TYPE: TYPE_LIVING}
export_map['tb_map'][PHYS_ACTIVITY] =   {STATUS: STATUS_TIME}
export_map['tb_map'][SEXUAL_ORIENT] =   {STATUS: STATUS_TIME, TYPE: TYPE_SEXUAL_ORIENT}


export_map['attr_map'] = {}
export_map['attr_map'][ALCOHOL] =           {STATUS: STATUS_TIME_VAL}
export_map['attr_map'][DRUG] =              {STATUS: STATUS_TIME_VAL}
export_map['attr_map'][TOBACCO] =           {STATUS: STATUS_TIME_VAL}
export_map['attr_map'][EMPLOYMENT] =        {STATUS: STATUS_EMPLOY_VAL}
export_map['attr_map'][ENVIRO_EXPOSURE] =   {STATUS: STATUS_TIME_VAL}
export_map['attr_map'][GENDER_ID] =         {STATUS: STATUS_TIME_VAL, TYPE: TYPE_GENDER_ID_VAL}
export_map['attr_map'][INSURANCE] =         {STATUS: STATUS_INSURE_VAL}
export_map['attr_map'][LIVING_STATUS] =     {STATUS: STATUS_TIME_VAL, TYPE: TYPE_LIVING_VAL}
export_map['attr_map'][PHYS_ACTIVITY] =     {STATUS: STATUS_TIME_VAL}
export_map['attr_map'][SEXUAL_ORIENT] =     {STATUS: STATUS_TIME_VAL, TYPE: TYPE_SEXUAL_ORIENT_VAL}
