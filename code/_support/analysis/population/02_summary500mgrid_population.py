"Stage 2: aggregate saved 100 m population results to existing 500 m grids by overlap area."

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter


sys.dont_write_bytecode = True

# Locate GIS libraries without reopening population or flood rasters.
runtime_root = Path(sys.executable).resolve().parent
for env_name, relative_path in [
    ("GDAL_DATA", "Library/share/gdal"),
    ("PROJ_DATA", "Library/share/proj"),
]:
    runtime_path = runtime_root / relative_path
    if runtime_path.exists():
        os.environ.setdefault(env_name, str(runtime_path))

import pandas as pd

from _support.analysis_paths import city_paths
from functions import aggregate_saved_pixels


def _read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, nargs="+", type=Path,
                        help="One or more stage-1 city batch directories.")
    args = parser.parse_args()
    for input_dir in args.input_dir:
        source_dir = input_dir.resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError()
        information_paths = list(source_dir.glob("*_100mspatial_metadata.json"))
        if len(information_paths) != 1:
            raise ValueError()
        information = _read_json(information_paths[0])
        # Reuse the batch settings and resolve its grid from the current package.
        cfg = information["config"]
        source_record = information["record"]
        city = cfg["CITY"]
        cfg["GRID_PATH"] = city_paths(city)["grid"]
        if source_record.get("status") != "completed" or source_record.get("stage") != "100m":
            raise ValueError()
        if source_record.get("city") != city:
            raise ValueError()
        source_csv = source_dir / information["pixel_csv"]
        if not source_csv.is_file():
            raise FileNotFoundError()
        csv_path = source_dir / f"{city}_500mgrid_population_and_flood_exposure.csv"
        if csv_path.exists():
            raise FileExistsError()

        timer = perf_counter()
        print(f"[{cfg['CITY'].title()}] Aggregating saved 100 m results to 500 m grids", flush=True)
        pixels = pd.read_csv(source_csv, encoding="utf-8-sig", dtype={"pixel_id": str})
        # Reconstruct geometry and overlap areas; population and exposure values come from the saved CSV.
        result, summary, _ = aggregate_saved_pixels(cfg, pixels, source_record)
        # Create only the new CSV exclusively, leaving stage-1 results read only.
        result.to_csv(csv_path, mode="x", index=False, encoding="utf-8-sig", na_rep="")
        print(f"[{cfg['CITY'].title()}] Known total population: {summary.iloc[0]['population']:,.3f}", flush=True)
        for row in summary.itertuples(index=False):
            print(f"  {row.scenario}: exposed population {row.population_exposed:,.3f}, "
                  f"population without inundation {row.population_nonexposed:,.3f}", flush=True)
        print(f"[{cfg['CITY'].title()}] Completed {len(result):,} grids in {perf_counter() - timer:.2f}s; "
              f"output: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
