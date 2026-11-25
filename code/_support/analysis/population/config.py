from pathlib import Path
import os
from _release import ROOT
from _support.analysis_paths import CITY_CODES, CITY_NAMES, EPSG, city_paths, config, configured_cities, output_root

PARAM = config("exposure_and_roads", "EXPOSURE_CONFIG")["population"]
CITIES = configured_cities("exposure_and_roads", "EXPOSURE_CONFIG", "city_order")
ANALYSIS_SCOPE = "city"
DISTRICT_NAME = PARAM["district"]
OUTPUT_DIR = output_root("population")

def cfg_for_city(city):
    source = city_paths(city)
    root = Path(os.environ.get("RELEASE_DEPTH_ROOT", ROOT/"results/flood_simulation"))
    periods = PARAM["periods"]
    return {"CITY":city,"CITY_NAME":CITY_NAMES[city],"ANALYSIS_SCOPE":ANALYSIS_SCOPE,
            "DISTRICT_NAME":DISTRICT_NAME,"GRID_PATH":source['grid'],"BOUNDARY_PATH":source['boundary'],
            "DISTRICT_PATH":source['districts'],"POPULATION_PATH":source['population_raster'],
            "FLOOD_PATHS":{f"T{p}":root/city/f"T{p:03d}_D12mmh/max_depth_land_study_area_m.tif" for p in periods},
            "DEPTH_THRESHOLD_M":PARAM["depth_threshold_m"],"GEOMETRY_BATCH_SIZE":PARAM["geometry_batch_size"]}
