"Stage 1: calculate population covered by depth >=0.30 m within complete original 100 m pixels."

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter


sys.dont_write_bytecode = True

# Locate GIS libraries in the current Conda environment without generating bytecode.
runtime_root = Path(sys.executable).resolve().parent
for env_name, relative_path in [
    ("GDAL_DATA", "Library/share/gdal"),
    ("PROJ_DATA", "Library/share/proj"),
]:
    runtime_path = runtime_root / relative_path
    if runtime_path.exists():
        os.environ.setdefault(env_name, str(runtime_path))

from config import ANALYSIS_SCOPE, CITIES, DISTRICT_NAME, OUTPUT_DIR, cfg_for_city
from functions import calculate_100m_pixels


CHINA_TIME = timezone(timedelta(hours=8), "Asia/Shanghai")


def _now():
    return datetime.now(CHINA_TIME)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError()


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  default=_json_default, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", default=CITIES,
                        help="City code; defaults to Fuzhou.")
    parser.add_argument("--scope", choices=["city", "district"], default=ANALYSIS_SCOPE,
                        help="Use all original grids in city mode or centroid-selected grids in district mode (default city).")
    parser.add_argument("--district", default=DISTRICT_NAME,
                        help="District name in district mode; defaults to Taijiang District.")
    parser.add_argument("--grid-limit", type=int, default=None,
                        help="Use only the first N original grids for a smoke check; defaults to all grids.")
    args = parser.parse_args()
    if args.grid_limit is not None and args.grid_limit <= 0:
        parser.error('')
    if len(set(args.cities)) != len(args.cities):
        parser.error('')
    if args.scope == "district" and not args.district.strip():
        parser.error('')

    configs = [cfg_for_city(city) for city in args.cities]
    for cfg in configs:
        cfg["ANALYSIS_SCOPE"] = args.scope
        cfg["DISTRICT_NAME"] = args.district.strip()
    started_at = _now()
    scope_name = "citywide" if args.scope == "city" else "district"
    if args.grid_limit is not None:
        scope_name += f"_first_{args.grid_limit}_grids"
    batch_name = f"{started_at:%Y%m%d_%H%M%S_%f}_{scope_name}"
    for cfg in configs:
        city = cfg["CITY"]
        # Use one batch per city; stage 2 appends its 500 m table to the same directory.
        batch_dir = OUTPUT_DIR / city / batch_name
        batch_dir.mkdir(parents=True, exist_ok=False)
        timer = perf_counter()
        print(f"[{city.title()}] Starting 100 m exposed-population calculation: {scope_name}", flush=True)
        pixels, record = calculate_100m_pixels(cfg, grid_limit=args.grid_limit)
        pixel_csv_path = batch_dir / f"{city}_100mpopulation_and_flood_exposure.csv"
        # population stores complete source-pixel population; use population_in_selected_grids for selected-area totals.
        pixels.to_csv(pixel_csv_path, index=False, encoding="utf-8-sig", na_rep="")
        record.update({
            "analysis_scope": args.scope,
            "district_name": cfg["DISTRICT_NAME"] if args.scope == "district" else None,
            "timezone": "Asia/Shanghai",
            "started_at": started_at.isoformat(),
            "completed_at": _now().isoformat(),
            "elapsed_seconds": round(perf_counter() - timer, 3),
        })
        # Write completion metadata only after the CSV has been fully saved.
        _write_json(batch_dir / f"{city}_100mspatial_metadata.json", {
            "config": cfg,
            "record": record,
            "pixel_csv": pixel_csv_path.name,
        })
        print(f"[{city.title()}] Completed {len(pixels):,} original 100 m pixels: {pixel_csv_path}", flush=True)
        print(f"Stage-2 input directory: {batch_dir}", flush=True)


if __name__ == "__main__":
    main()
