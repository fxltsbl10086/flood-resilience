"Load original unweighted 500 m SAR grids for the five manuscript cities."

import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

# Initialize the Windows GIS runtime when applicable.
_dll_handles = []
_runtime_bin = Path(sys.prefix) / "Library/bin"
if os.name == "nt" and _runtime_bin.is_dir():
    os.environ["PATH"] = str(_runtime_bin) + os.pathsep + os.environ.get("PATH", "")
    _dll_handles.append(os.add_dll_directory(str(_runtime_bin)))
for _variable, _relative in (("GDAL_DATA", "Library/share/gdal"),
                             ("PROJ_DATA", "Library/share/proj")):
    if (Path(sys.prefix) / _relative).is_dir():
        os.environ.setdefault(_variable, str(Path(sys.prefix) / _relative))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from sar_config import (
    OUTPUT_DIR, PERIODS, CITY_ORDER, CITY_NAMES, CITY_ABBR, CITY_INPUTS,
    RISK_COLUMN, FIGURE_MM, ROWS, COLUMNS, DPI, MAP_LAYOUT_MM,
    COLORBAR_WIDTH_MM, COLORBAR_BOTTOM_MM, COLORBAR_HEIGHT_MM,
    FONT_FILE, BODY_FONT_SIZE, AXIS_FONT_SIZE,
    CITY_FONT_SIZE, PANEL_FONT_SIZE, LEGEND_FONT_SIZE, MAP_FRAME_LINEWIDTH,
    STUDY_BOUNDARY_LINEWIDTH, TICK_LENGTH, TICK_PAD, PANEL_LABEL_OFFSET,
    CITY_LABEL_OFFSET, COORDINATE_TICK_INSETS, EXTENT_FILL_FRACTION,
    RISK_COLORS, RISK_LIMITS, EXCLUDED_COLOR,
)


def read_flag(values):
    "Accept existing boolean flag encodings and stop on missing flags."
    result = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    if result.isna().any():
        raise ValueError()
    return result.astype(bool)


def load_data(periods):
    "Read original five-city grids and SAR without population filtering or model reruns."
    datasets = {}
    risk_columns = [RISK_COLUMN.format(period=period) for period in periods]
    fields = ["grid_id", "centroid_x", "centroid_y", "baseline_valid", *risk_columns]
    for city in CITY_ORDER:
        cfg = CITY_INPUTS[city]
        table = pd.read_csv(cfg["RISK_CSV"], encoding="utf-8-sig",
                            dtype={"grid_id": str}, usecols=fields)
        grids = gpd.read_file(cfg["GRID_PATH"])[["grid_id", "geometry"]]
        boundary = gpd.read_file(cfg["BOUNDARY_PATH"])
        grids["grid_id"] = grids["grid_id"].astype(str)
        if (table.grid_id.isna().any() or not table.grid_id.is_unique
                or not grids.grid_id.is_unique or set(table.grid_id) != set(grids.grid_id)):
            raise ValueError()
        if grids.crs is None or boundary.crs is None or not grids.crs.is_projected:
            raise ValueError()
        data = grids.merge(table, on="grid_id", how="left", validate="one_to_one")
        centers = data.geometry.centroid
        if not (np.allclose(centers.x, data.centroid_x, atol=1e-5, rtol=0)
                and np.allclose(centers.y, data.centroid_y, atol=1e-5, rtol=0)):
            raise ValueError()

        # Use the same baseline-valid grid set for all five scenarios.
        included = read_flag(data.baseline_valid)
        if not included.any():
            raise ValueError()
        risk = data.loc[included, risk_columns].to_numpy(float)
        if not np.isfinite(risk).all() or np.any((risk < RISK_LIMITS[0]) | (risk > RISK_LIMITS[1])):
            raise ValueError()
        data["analysis_included"] = included
        datasets[city] = (data.to_crs(4326), boundary.to_crs(4326))
        print(f"{CITY_NAMES[city]}: {int(included.sum())}/{len(data)} baseline-valid grids included in unweighted SAR.", flush=True)
    return datasets


def set_style():
    "Set the configured font and white-background framed axes."
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        
        if FONT_FILE.with_name(name).is_file():
            font_manager.fontManager.addfont(str(FONT_FILE.with_name(name)))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(FONT_FILE)).get_name(),
        "font.size": BODY_FONT_SIZE, "axes.unicode_minus": False,
        "axes.linewidth": MAP_FRAME_LINEWIDTH, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


