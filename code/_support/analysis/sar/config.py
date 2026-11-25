import os
from _release import ROOT
from _support.analysis_paths import CITY_CODES, CITY_NAMES, EPSG, city_paths, config, city_selection, output_root

PARAM = config("accessibility_risk", "ACCESS_CONFIG")
SRC = PARAM['sar']
PROJECT_ROOT = ROOT
SOURCE_ROOT = ROOT
CITY = os.environ.get("RELEASE_CITY") or city_selection(PARAM["cities"])[0]
PERIODS = PARAM['return_periods']

def config_for_city(city):
    f = city_paths(city)
    return {
        "ROOT_MAIN":ROOT,"CITY":city,"PERIODS":PERIODS,"ORIGIN_SCOPE":SRC['origin_scope'],"DISTRICT_NAME":"台江区",
        "GRID_PATH":f['grid'],"BOUNDARY_PATH":f['boundary'],"DISTRICT_PATH":f['districts'],"RAW_ROAD_PATH":f['raw_roads'],
        "ROAD_PATH":f['road_status'],"EDGE_PATH":f['edge_status'],"ROAD_LAYER":"road_flood_status","EDGE_LAYER":"routing_edge_flood_status",
        "ACCESS_EXCLUDED_CLASSES":SRC['excluded_road_types'],"MAX_ACCESS_DISTANCE_M":SRC['max_access_distance_m'],
        "SOURCE_GEOMETRY_TOLERANCE_M":SRC['source_geometry_tolerance_m'],"TEST_ORIGIN_LIMIT":SRC['test_origin_limit'],
        "RUN_FULL_AFTER_TEST":os.environ.get('RELEASE_SAR_TEST_ONLY')!='1',"BATCH_SIZE":SRC['batch_size'],
        "DISTANCE_TOLERANCE_M":SRC['distance_tolerance_m'],"RISK_TOLERANCE":SRC['risk_tolerance'],"OUTPUT_DIR":output_root('sar'),
    }

CONFIG = config_for_city(CITY)
