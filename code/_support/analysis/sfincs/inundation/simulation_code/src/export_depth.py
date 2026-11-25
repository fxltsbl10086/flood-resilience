"""Export native SFINCS maximum and final depths without changing hydrodynamic results."""
from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import xarray as xr

SFINCS = Path(__file__).resolve().parents[3]
# Locate the release and input modules independently of the working directory.
INPUT_CODE = SFINCS / "inputs"
CODE = SFINCS.parents[2]
for directory in (CODE, INPUT_CODE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from _release import package_path
from build_inputs import load_config
from sfincs_io import read_static
from spatial import get_landuse_fractions, read_landuse_fractions_cache


DEPTH_PRODUCTS = (
    ("max_depth_domain_m.tif", "maximum", "domain", "Full active computational domain"),
    ("max_depth_study_area_m.tif", "maximum", "study_area", "Study area including water"),
    ("max_depth_land_study_area_m.tif", "maximum", "land_study_area", "Study-area land"),
    ("final_depth_domain_m.tif", "final", "domain", "Full active computational domain"),
    ("final_depth_study_area_m.tif", "final", "study_area", "Study area including water"),
    ("final_depth_land_study_area_m.tif", "final", "land_study_area", "Study-area land"),
)


def write_depth(path, values, selected, grid, *, city, period, product, scope, start, stop):
    """Apply spatial masks and convert formats without smoothing or patch removal."""
    data = np.where(selected, values, -9999).astype("float32")
    if product == "maximum":
        description = f"Maximum depth from {start} to {stop}, in m; {scope}"
    else:
        description = f"Depth at the simulation end {stop}, in m; {scope}"
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="float32", crs=grid["crs"], transform=grid["transform"],
                       nodata=-9999, tiled=True, compress="deflate") as destination:
        destination.write(data, 1)
        destination.set_band_description(1, description)
        destination.update_tags(
            city=city, return_period_years=period, units="m", source="sfincs_map.nc",
            product=product, scope=scope, simulation_start=start, simulation_stop=stop,
            event_peak_convergence_verified="false", spatial_smoothing="none",
            nodata_interpretation="Outside the selected spatial extent; not equivalent to zero depth",
            zero_interpretation=("Native hmax fill values below the output threshold are displayed as zero; zero does not establish exact dryness"
                                 if product == "maximum" else
                                 "Negative native final depths are displayed as zero only in GIS; native NetCDF is unchanged"),
            hmax_output_threshold_m=grid["config"].get("huthresh", "未记录"),
        )


def _time_text(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _result_directory(results, period):
    """Identify one scenario by its T prefix without fixing the drainage label."""
    if not results.is_dir():
        raise FileNotFoundError()
    prefix = re.compile(rf"^T{period:03d}(?:_|$)", re.IGNORECASE)
    candidates = sorted(path for path in results.iterdir()
                        if path.is_dir() and prefix.match(path.name))
    if not candidates:
        raise FileNotFoundError()
    if len(candidates) != 1:
        raise ValueError()
    return candidates[0]


def _model_input(root, scenario):
    """Locate the solver input within the city directory."""
    path = root / scenario / "sfincs.inp"
    if not path.is_file():
        raise FileNotFoundError()
    return path


def _read_depths(path, grid, scenario):
    """Validate completed native outputs against inputs, then read all hmax windows and final h."""
    active = grid["mask"] > 0
    # Use HDF5-compatible reading for Windows paths containing non-ASCII characters.
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        if "time" not in ds or not ds.time.size:
            raise ValueError()
        start = pd.to_datetime(grid["config"]["tstart"], format="%Y%m%d %H%M%S")
        stop = pd.to_datetime(grid["config"]["tstop"], format="%Y%m%d %H%M%S")
        if pd.Timestamp(ds.time.values[-1]) != stop or float(ds.status.values[-1]) != 0:
            raise ValueError()
        if not np.array_equal(np.flipud(ds.msk.values), grid["mask"]):
            raise ValueError()
        if not np.allclose(np.flipud(ds.zb.values)[active], grid["terrain"][active],
                           rtol=0, atol=1e-5):
            raise ValueError()
        # The solver resets hmax per window; maximize across all timemax windows.
        if "timemax" not in ds.hmax.dims or not ds.timemax.size:
            raise ValueError()
        native_max = np.flipud(ds.hmax.max("timemax", skipna=True).values)
        native_final = np.flipud(ds.h.isel(time=-1).values)
        if np.isnan(native_final[active]).any():
            count = int(np.count_nonzero(np.isnan(native_final[active])))
            raise ValueError()
        if np.isinf(native_max[active]).any() or np.isinf(native_final[active]).any():
            raise ValueError()
        # Display hmax fill values as zero; require finite active final depths and clip negatives only for GIS.
        maximum = np.maximum(np.nan_to_num(native_max, nan=0), 0).astype("float32")
        final = np.maximum(native_final, 0).astype("float32")
    return {"maximum": maximum, "final": final}, _time_text(start), _time_text(stop)


def export_depth(city, model_root, result_root, periods, overwrite=False):
    """Export six depth GeoTIFFs per scenario, reusing existing files individually.

model_root/result_root select city directories; None uses the default layout.
Return written and reused files without changing historical statistics or native NetCDF."""
    config = load_config(city)
    root = Path(model_root) if model_root is not None else package_path(config["model_inputs"]) / city
    results = (Path(result_root) if result_root is not None else
               package_path(config["archive"]) / "recomputed" / city)
    selected_periods = sorted(set(periods))
    if not selected_periods or any(p not in config["return_periods"] for p in selected_periods):
        raise ValueError()
    grid, masks = None, None
    records = []
    for period in selected_periods:
        directory = _result_directory(results, period)
        scenario = directory.name
        targets = {name: directory / name for name, _, _, _ in DEPTH_PRODUCTS}
        reused = [name for name, path in targets.items() if path.is_file() and not overwrite]
        missing = [name for name in targets if name not in reused]
        if any(path.exists() and not path.is_file() for path in targets.values()):
            raise FileExistsError()
        record = {"city": city, "scenario": scenario, "return_period_years": period,
                  "files": {name: str(path.resolve()) for name, path in targets.items()},
                  "reused_files": reused, "exported_files": []}
        if not missing:
            print(f"[{city}] {scenario} Reusing all six existing depth GeoTIFFs; use --overwrite to re-export")
            records.append(record)
            continue
        current = read_static(_model_input(root, scenario))
        if grid is None:
            grid = current
            active = grid["mask"] > 0
            study = gpd.read_file(config["sources"]["study_area"]).to_crs(grid["crs"])
            study_cells = geometry_mask(study.geometry, grid["mask"].shape, grid["transform"], invert=True)
            # Coverage diagnostics require study-area geometry, which static binaries do not store.
            grid["study"] = study.geometry.union_all()
            landuse = config["sources"]["landuse"]
            try:
                fractions = read_landuse_fractions_cache(root / "landuse_fractions.tif", landuse, grid)
            except (OSError, ValueError, rasterio.errors.RasterioError):
                fractions = None
            if fractions is None:
                fractions = get_landuse_fractions(landuse, grid, results)
            water = fractions[np.array(config["water_codes"]) - 1].sum(axis=0) >= config["water_fraction_threshold"]
            # Exclude outer outflow cells from study-area products and thresholded water cells from land products.
            study_cells &= grid["mask"] == 1
            masks = {"domain": active, "study_area": study_cells,
                     "land_study_area": study_cells & ~water}
        elif (current["transform"] != grid["transform"] or current["crs"] != grid["crs"]
              or not np.array_equal(current["mask"], grid["mask"])
              or not np.allclose(current["terrain"][grid["mask"] > 0],
                                 grid["terrain"][grid["mask"] > 0], rtol=0, atol=1e-5)):
            raise ValueError()
        depths, start, stop = _read_depths(directory / "sfincs_map.nc", current, scenario)
        for name, product, mask_name, scope in DEPTH_PRODUCTS:
            if name in reused:
                continue
            write_depth(targets[name], depths[product], masks[mask_name], current,
                        city=city, period=period, product=product, scope=scope, start=start, stop=stop)
            record["exported_files"].append(name)
        record.update(simulation_start=start, simulation_stop=stop)
        records.append(record)
        print(f"[{city}] {scenario} Depth GeoTIFFs: {len(missing)} written, {len(reused)} reused")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True)
    parser.add_argument("--model-root", type=Path, help="Default: results/flood_simulation/model_inputs/<city>")
    parser.add_argument("--result-root", type=Path, help="Default: results/flood_simulation/recomputed/<city>")
    parser.add_argument("--return-periods", type=int, nargs="+", default=[10, 20, 50, 100, 200])
    parser.add_argument("--overwrite", action="store_true", help="Re-export existing GeoTIFFs without rerunning hydrodynamics")
    args = parser.parse_args()
    export_depth(args.city, args.model_root, args.result_root, args.return_periods, args.overwrite)


if __name__ == "__main__":
    main()
