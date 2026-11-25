from _support.analysis_paths import CITY_CODES, CITY_NAMES, city_paths, config, configured_cities, output_root

CITIES = configured_cities("accessibility_risk", "ACCESS_CONFIG")
PERIODS = config("accessibility_risk", "ACCESS_CONFIG")['return_periods']

def config_for_city(city):
    f=city_paths(city)
    return {"CITY":city,"CITY_NAME":CITY_NAMES[city],"ANALYSIS_AREA":"全研究区","ANALYSIS_AREA_DIR":"full_study_area",
            "POPULATION_CSV":f["pop"],"SAR_CSV":f["sar"],"GRID_IDS":None,
            "PERIODS":PERIODS,"OUTPUT_DIR":output_root("weighted_sar"),
            "COORDINATE_TOLERANCE_M":1e-5,"POPULATION_TOLERANCE":1e-6,"SAR_TOLERANCE":1e-10}
