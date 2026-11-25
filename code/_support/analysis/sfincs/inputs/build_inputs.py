"""Build SFINCS inputs for one or more cities without running the solver."""
from __future__ import annotations

import argparse
import os
from _release import load_config as load_release_config, package_path
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from hydrology import read_rainfall, runoff_frames
from spatial import get_landuse_fractions, load_grid, rainfall_zones
from sfincs_io import write_static, write_forcing, write_input

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]


def load_config(city):
    _, config = load_release_config("flood_simulation", os.environ.get("FLOOD_CONFIG"))
    settings = config.pop("cities")[city]
    config.update(settings)
    config["sources"] = {key: package_path(value) for key,value in config["sources"].items()}
    config["rainfall_table"] = package_path(config["rainfall_table"])
    config["cn_table"] = package_path(config["cn_table"])
    return config


def build_city(city, periods=None, rebuild=False):
    """Write solver inputs and validated eight-class fractions for rebuilding and depth export."""
    config = load_config(city)
    selected = sorted(set(periods or config["return_periods"]))
    if not selected or any(period not in config["return_periods"] for period in selected):
        raise ValueError()
    root = Path(os.environ["FLOOD_MODEL_OUTPUT"]) / city
    # Reuse existing inputs unless --rebuild explicitly replaces the city input set.
    if root.exists() and any(root.iterdir()):
        if rebuild:
            if selected != sorted(config["return_periods"]):
                raise ValueError()
        else:
            required = [root / name for name in ("sfincs.ind", "sfincs.dep", "sfincs.msk",
                                                 "sfincs.man", "sfincs.qinf")]
            required += [root / f"T{p:03d}_D12mmh" / name for p in selected
                         for name in ("sfincs.inp", "sfincs_cn_runoff.nc")]
            if not all(path.is_file() for path in required):
                raise FileExistsError()
            print(f"[{city}] Reusing inputs: {root}; use --rebuild after parameter changes")
            return
    for path in [*config["sources"].values(), config["rainfall_table"], config["cn_table"]]:
        if not path.is_file():
            raise FileNotFoundError()
    rain = read_rainfall(config, selected)
    grid = load_grid(config)
    print(f"[{city}] Grid {grid['mask'].shape}, active cells {np.count_nonzero(grid['mask'])}", flush=True)
    fractions = get_landuse_fractions(config["sources"]["landuse"], grid, root)
    zones = rainfall_zones(config, grid, tuple(rain[selected[0]]))
    table = pd.read_csv(config["cn_table"]).set_index("landuse_code").sort_index()
    if list(table.index) != list(range(1, 9)):
        raise ValueError()
    cn = table[f"cn_{config['soil_group'].lower()}"].to_numpy(dtype=float)
    if not np.all((cn > 0) & (cn <= 100)):
        raise ValueError()
    # Built surfaces are other impervious cover, buildings and roads; drainage is not deducted during CN runoff.
    drainage = config["drainage_mm_h"] * fractions[np.array(config["drainage_codes"]) - 1].sum(axis=0)
    root.mkdir(parents=True, exist_ok=True)
    write_static(root, grid, drainage, config["manning_n"])
    for period in selected:
        directory = root / f"T{period:03d}_D12mmh"
        directory.mkdir(exist_ok=True)
        frames = runoff_frames(config, fractions, cn, rain[period], zones, grid["mask"] > 0)
        write_forcing(directory / "sfincs_cn_runoff.nc", grid, config, frames)
        write_input(directory / "sfincs.inp", grid, config)
        print(f"[{city}] T{period:03d} Inputs complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--cities", nargs="+", help="For example: fuzhou, or fuzhou wuhan hangzhou")
    choice.add_argument("--all", action="store_true", help="Process every configured city")
    parser.add_argument("--return-periods", type=int, nargs="+", help="Build only the selected return periods for new cities")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all city inputs without rerunning existing depth results")
    args = parser.parse_args()
    cities = args.cities or list(load_release_config("flood_simulation", os.environ.get("FLOOD_CONFIG"))[1]["cities"])
    failed = []
    for city in dict.fromkeys(cities):
        try:
            build_city(city, args.return_periods, args.rebuild)
        except Exception as error:
            None
            failed.append(city)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
