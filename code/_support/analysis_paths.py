"""Locate frozen analysis inputs independently of the original workspace."""
import json
import os
from functools import lru_cache
from pathlib import Path

from _release import ANALYSIS, CODE, ROOT, package_path, load_config

CITY_CODES = ["chengdu", "fuzhou", "hangzhou", "wuhan", "zhengzhou"]
CITY_NAMES = {"chengdu": "成都", "fuzhou": "福州", "hangzhou": "杭州",
              "wuhan": "武汉", "zhengzhou": "郑州"}
EPSG = {"chengdu": 32648, "fuzhou": 32650, "hangzhou": 32651,
        "wuhan": 32650, "zhengzhou": 32649}


@lru_cache(maxsize=1)
def batches():
    return json.loads((ANALYSIS / "city_batches.json").read_text(encoding="utf-8"))


def config(section, variable):
    return load_config(section, os.environ.get(variable))[1]


def city_selection(values):
    cities = list(values)
    if not cities or len(cities) != len(set(cities)) or not set(cities).issubset(CITY_CODES):
        raise ValueError()
    return cities


def configured_cities(section, variable, key="cities"):
    return city_selection(config(section, variable)[key])


def output_root(name=None):
    directory = Path(os.environ.get("RELEASE_ANALYSIS_OUTPUT", ANALYSIS / "recomputed"))
    return directory / name if name else directory


def city_paths(city):
    directory = ANALYSIS / "prepared" / city
    road = ANALYSIS / "road_status" / city
    return {
        "grid": package_path(batches()[city]["grid"]),
        "boundary": directory / "boundary.shp",
        "districts": directory / "districts.shp",
        "raw_roads": directory / "roads.shp",
        "population_raster": directory / "population.tif",
        "road_status": road / "01-01__road_flood_status_five_scenarios.gpkg",
        "edge_status": road / "01-03__directed_edge_flood_status_five_scenarios.gpkg",
        "pop": ANALYSIS / "population" / city / f"{city}_500mgrid_population_and_flood_exposure.csv",
        "sar": ANALYSIS / "sar" / city / "grid_risk.csv",
        "efar": ANALYSIS / "efar" / city / f"{city}_three_function_accessibility_risk.csv",
        "efar_metadata": ANALYSIS / "efar" / city / "risk_calculation_methods.json",
        "counts": ANALYSIS / "facility_counts" / city / f"{city}_grid_facility_counts.csv",
    }


def access_inputs(city):
    paths = city_paths(city)
    return {
        "population_csv": paths["pop"],
        "risk_csv": paths["efar"],
        "risk_metadata": paths["efar_metadata"],
    }
