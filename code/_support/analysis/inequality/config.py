from _support.analysis_paths import CITY_CODES, CITY_NAMES as CITY_NAMES_ZH, access_inputs
from _support.analysis_paths import config, configured_cities
CITIES=configured_cities("accessibility_risk", "ACCESS_CONFIG")
CITY_NAMES={city:city.capitalize() for city in CITY_CODES}
INPUTS={city:access_inputs(city) for city in CITIES}
PERIODS=config('accessibility_risk','ACCESS_CONFIG')['return_periods']
SERVICE_DIMENSIONS=['应急救援','医疗资源','避险转移']
DIMENSIONS=['exposure',*SERVICE_DIMENSIONS]
DIMENSION_NAMES=dict(zip(DIMENSIONS,['Exposure','Emergency rescue','Healthcare','Emergency shelter']))
