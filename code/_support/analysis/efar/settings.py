"Three-function resource inputs using the existing roads and 500 m grids."
from pathlib import Path
import importlib.util
import os
import sys

# Use GDAL data from the current Python environment.
gdal_data = Path(sys.prefix) / 'Library/share/gdal'
if gdal_data.is_dir():
    os.environ.setdefault('GDAL_DATA', str(gdal_data))

HERE = Path(__file__).resolve().parent
from _release import ROOT
from _support.analysis_paths import configured_cities
PROJECT = ROOT
ENGINE = HERE.parent / "sar"
BASE_POI = PROJECT / "data/raw/source_facilities"
SUPPLEMENT = PROJECT / "data/raw/source_shelter_registers"
CITIES = {'fuzhou': '福州', 'chengdu': '成都', 'wuhan': '武汉',
          'hangzhou': '杭州', 'zhengzhou': '郑州'}
DEFAULT_CITIES = configured_cities("accessibility_risk", "ACCESS_CONFIG")
DIMENSIONS = ['应急救援', '医疗资源', '避险转移']
DIRECTIONS = ['设施→居民', '居民→设施', '居民→设施']
PERIODS = [10, 20, 50, 100, 200]

def scope_name(cities):
    if set(cities) == set(CITIES):
        return 'five_cities'
    return cities[0] if len(cities) == 1 else 'selected_cities'

def load_engine(name):
    spec = importlib.util.spec_from_file_location('resource_' + name, ENGINE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

engine_config = load_engine('config')
input_functions = load_engine('functions')
routing = load_engine('reachability')

def city_config(city):
    return engine_config.config_for_city(city)
