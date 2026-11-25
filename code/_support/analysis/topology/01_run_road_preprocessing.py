"Road-preprocessing entry with explicit per-city parameters."

from __future__ import annotations

from pathlib import Path

from config import load_config
from src.road_processing import expected_output_paths, run_city


import os
import sys
from _release import CODE
from _support.analysis_paths import CITY_CODES, EPSG, city_paths, configured_cities, output_root

COMMON_CONFIG_PATH = Path(os.environ.get("EXPOSURE_CONFIG", CODE / "exposure_and_roads.yaml"))
OUTPUT_ROOT=output_root("road_topology")
CITIES_TO_RUN=sys.argv[1:] or configured_cities("exposure_and_roads", "EXPOSURE_CONFIG", "city_order")

def main() -> int:
    # Deduplicate city codes while retaining their configured order.
    city_names = list(dict.fromkeys(CITIES_TO_RUN))
    if not city_names:
        raise ValueError()
    unknown_cities = [name for name in city_names if name not in CITY_CODES]
    if unknown_cities:
        raise ValueError()
    configs = [
        load_config(COMMON_CONFIG_PATH, {"city_code": name,
            "road": city_paths(name)["raw_roads"], "boundary": city_paths(name)["boundary"],
            "target_crs": f"EPSG:{EPSG[name]}"})
        for name in city_names
    ]

    # Check all output destinations before starting the batch.
    existing = [
        path
        for config in configs
        for path in expected_output_paths(OUTPUT_ROOT, config.city_code).values()
        if path.exists()
    ]
    if existing:
        raise FileExistsError()

    for config in configs:
        print(f"Starting: {config.city_code}")
        result = run_city(config, OUTPUT_ROOT)
        print(
            f"Completed: {result['city_code']}, "
            f"facility segments {result['facility_count']}, "
            f"nodes {result['node_count']}, "
            f"directed edges {result['edge_count']}, "
            f"output directory {result['output_dir']}"
        )
    return 0


if __name__ == "__main__":
    main()
