from _support.analysis_paths import CITY_CODES, CITY_NAMES, access_inputs
from _support.analysis_paths import config, configured_cities
CITY_ORDER=configured_cities("accessibility_risk", "ACCESS_CONFIG")
CITY_INPUTS={city:access_inputs(city) for city in CITY_ORDER}
CITY_ABBR={'chengdu':'CD','zhengzhou':'ZZ','hangzhou':'HZ','wuhan':'WH','fuzhou':'FZ'}
PERIODS=config('accessibility_risk','ACCESS_CONFIG')['return_periods']
SERVICE_DIMENSIONS=['应急救援','医疗资源','避险转移']
