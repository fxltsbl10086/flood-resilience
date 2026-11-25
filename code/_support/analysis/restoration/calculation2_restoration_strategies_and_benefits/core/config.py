"""Road-restoration priorities for five cities and five return periods; each subprocess handles one city and scenario."""
from pathlib import Path
import os
from _release import ROOT, package_path
from _support.analysis_paths import config as release_config
import sys

HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE
DLL_HANDLES = []
runtime_bin = Path(sys.prefix) / "Library/bin"
if os.name == "nt" and runtime_bin.is_dir():
    os.environ["PATH"] = str(runtime_bin) + os.pathsep + os.environ.get("PATH", "")
    DLL_HANDLES.append(os.add_dll_directory(str(runtime_bin)))
gdal_data = Path(sys.prefix) / "Library/share/gdal"
if gdal_data.is_dir():
    os.environ.setdefault("GDAL_DATA", str(gdal_data))

SOURCE_PROJECT = ROOT
_PARAMETERS = release_config('road_restoration','RESTORATION_CONFIG')
CITY = 'fuzhou'
SUPPORTED_CITIES = ('zhengzhou','hangzhou','fuzhou','chengdu','wuhan')
CITY_NAMES = {'zhengzhou':'郑州','hangzhou':'杭州','fuzhou':'福州','chengdu':'成都','wuhan':'武汉'}
PERIODS = tuple(_PARAMETERS['return_periods'])
EVALUATION_SCOPE = 'study_area'
DISTRICT_NAME = '台江区'
PERIOD = 10
EFARB_CSV = package_path(_PARAMETERS['baseline_burden_wide'])

COVERAGE_PERCENTAGES = list(range(0, 101, 5))
DIMENSION_KEYS = ["rescue", "medical", "shelter"]
DIMENSION_NAMES = {"rescue": "应急救援", "medical": "医疗资源", "shelter": "避险转移"}
LOCAL_DISTANCE_M = _PARAMETERS['same_grid_distance_m']
RISK_ATOL = 1e-10
DISTANCE_ATOL_M = 1e-6
BURDEN_ATOL = 1e-6
GEOMETRY_TOLERANCE_M = 1e-6
OUTPUT_ROOT = HERE / "outputs"
CACHE_ROOT = HERE / "cache"
CACHE_SCHEMA_VERSION = 2

# The fixed facility set is baseline-reachable under original EFAR; no facility-failure assumption is added.
FACILITY_RULE = "fixed_baseline_reachable_all_original_facilities"
# Keep evaluation facilities fixed; change only which target contacts define restoration projects.
GOVERNANCE_STRATEGY = "max_initial_EFAR_loss_facility_grid_contact_per_class_v1"
PRIORITY_ORDER = ["initial_EFARB:descending", "effective_population:descending",
                  "population_exposed:descending", "grid_id:ascending"]
# Without reliable speed coverage, do not convert distances to time using guessed speeds.
TRAVEL_TIME_ENABLED = False
